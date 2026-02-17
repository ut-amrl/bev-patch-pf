import argparse
import gc
import os
import random
from collections import OrderedDict

import hydra
import onnxruntime as ort
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

import onnx
from utils.misc import load_checkpoint, seed_everything

Hgnd, Wgnd = 518, 518  # ground image resolution
Haer, Waer = 768, 768  # aerial image resolution
N = 128  # number of hypotheses for aerial patch sampling


# ========== Wrappers for sub-modules ==========
class AerialImageEncoder(nn.Module):
    def __init__(self, aer_backbone: nn.Module, aer_upernet: nn.Module, aer_head: nn.Module):
        super().__init__()
        self.aer_backbone = aer_backbone
        self.aer_upernet = aer_upernet
        self.aer_head = aer_head

        # NOTE: freeze PPM for ONNX export
        self.aer_upernet.freeze_ppm_for_export(Haer // 32, Waer // 32)

    @torch.inference_mode()
    def forward(self, aer_img: torch.Tensor):
        aer_embs = self.aer_backbone(aer_img)
        aer_emb = self.aer_upernet(aer_embs)
        aer_feat = self.aer_head(aer_emb)
        return aer_feat


# ========== ONNX export & parity test utils ==========
def make_cuda_session(onnx_path: str):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider not available")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(onnx_path, sess_options=so, providers=["CUDAExecutionProvider"])


@torch.no_grad()
def run_ort(sess: ort.InferenceSession, inputs: dict[str, torch.Tensor]):
    outs = sess.run(None, {k: v.detach().cpu().numpy() for k, v in inputs.items()})
    return tuple(torch.from_numpy(o) for o in outs)


@torch.no_grad()
def run_pt(module: nn.Module, inputs: dict[str, torch.Tensor]):
    outs = module(*inputs.values())
    return outs if isinstance(outs, tuple) else (outs,)


def randomize_input(base: dict[str, torch.Tensor]):
    rnd = {}
    for name, t in base.items():
        if t.dtype == torch.bool:
            rnd[name] = torch.rand_like(t.float()) > 0.5
        else:
            rnd[name] = torch.randn_like(t)
    return rnd


def parity_stress(
    onnx_path: str,
    module: nn.Module,
    base_inputs: dict[str, torch.Tensor],
    iters: int,
    atol: float,
    rtol: float,
    tag: str,
):
    sess = make_cuda_session(onnx_path)
    for _ in range(2):  # warmup
        _ = run_ort(sess, base_inputs)

    worst = 0.0
    for i in tqdm(range(iters), desc=f"[{tag}] stress test", leave=False):
        seed = 123456 + i
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        inputs = randomize_input(base_inputs)
        ort_outs = run_ort(sess, inputs)
        pt_outs = run_pt(module, inputs)

        for j, (o_t, p_t) in enumerate(zip(ort_outs, pt_outs)):
            p_t = p_t.detach().cpu()
            if o_t.dtype.is_floating_point:
                o_t = o_t.to(torch.float32)
                p_t = p_t.to(torch.float32)
            assert o_t.shape == p_t.shape, f"[{tag}@{j}] {o_t.shape} != {p_t.shape}"
            if o_t.dtype == torch.bool:
                assert torch.equal(o_t, p_t), f"[{tag}@{j}] bool mismatch"
                continue
            if not torch.isfinite(o_t).all():
                raise AssertionError(f"[{tag}] ORT out{j} NaN/Inf")
            if not torch.isfinite(p_t).all():
                raise AssertionError(f"[{tag}] TORCH out{j} NaN/Inf")
            diff = torch.max(torch.abs(o_t - p_t)).item()
            worst = max(worst, diff)
            if not torch.allclose(o_t, p_t, atol=atol, rtol=rtol):
                raise AssertionError(f"[{tag}@{j}] max|diff|={diff:.6f} (atol={atol}, rtol={rtol})")

    print(f"✅ [{tag}] Stress OK: worst max|diff|={worst:.6f}")
    del sess, ort_outs, pt_outs
    torch.cuda.empty_cache()
    gc.collect()


# ========== Main export function ==========
def export_onnx(args):
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = hydra.utils.instantiate(OmegaConf.load(args.model_config))
    load_checkpoint(args.ckpt_path, model)
    model = model.cuda().eval()

    OUT_DIM = model.out_dim
    ENC_DIM = model.gnd_encoder.model.config.hidden_size
    BEV_DIM = model.bev_dim
    Pg = model.gnd_encoder.model.config.patch_size
    Pa = 4
    Hbev, Wbev = model.bev_mapper.bev_shape

    # ----- 1) ground image encoder -----
    gnd_encoder_path = f"{args.out_dir}/gnd_encoder.onnx"
    gnd_encoder_inputs = OrderedDict([("gnd_img", torch.rand(1, 3, Hgnd, Wgnd, device="cuda", dtype=torch.float32))])

    torch.onnx.export(
        model.gnd_encoder,
        tuple(gnd_encoder_inputs.values()),
        gnd_encoder_path,
        opset_version=20,
        input_names=list(gnd_encoder_inputs.keys()),
        output_names=["gnd_emb"],
        dynamic_axes={},
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx.checker.check_model(gnd_encoder_path)

    parity_stress(
        gnd_encoder_path,
        model.gnd_encoder,
        gnd_encoder_inputs,
        iters=args.iters,
        atol=args.atol,
        rtol=args.rtol,
        tag="gnd_encoder",
    )

    # ----- 2) BEV Mapper -----
    mapper_path = f"{args.out_dir}/bev_mapper.onnx"
    mapper_inputs = OrderedDict(
        [
            ("gnd_emb", torch.rand(1, ENC_DIM, Hgnd // Pg, Wgnd // Pg, dtype=torch.float32).cuda()),
            ("depth", torch.rand(1, Hgnd, Wgnd, dtype=torch.float32).cuda()),
            ("K", torch.eye(3, dtype=torch.float32).unsqueeze(0).cuda()),
            ("cam2enu", torch.eye(4, dtype=torch.float32).unsqueeze(0).cuda()),
            ("resolution", torch.tensor(0.3, dtype=torch.float32).unsqueeze(0).cuda()),
        ]
    )

    torch.onnx.export(
        model.bev_mapper,
        tuple(mapper_inputs.values()),
        mapper_path,
        opset_version=20,
        input_names=list(mapper_inputs.keys()),
        output_names=["bev_emb", "bev_mask"],
        dynamic_axes={"depth": {1: "H", 2: "W"}},
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx.checker.check_model(mapper_path)

    parity_stress(
        mapper_path,
        model.bev_mapper,
        mapper_inputs,
        iters=args.iters,
        atol=args.atol,
        rtol=args.rtol,
        tag="bev_mapper",
    )

    # ----- 3) BEV Encoder -----
    bev_encoder_path = f"{args.out_dir}/bev_encoder.onnx"
    bev_encoder_inputs = OrderedDict([("bev_emb", torch.rand(1, BEV_DIM, Hbev, Wbev, dtype=torch.float32).cuda())])

    torch.onnx.export(
        model.bev_encoder,
        tuple(bev_encoder_inputs.values()),
        bev_encoder_path,
        opset_version=20,
        input_names=list(bev_encoder_inputs.keys()),
        output_names=["bev_feat"],
        dynamic_axes={},
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx.checker.check_model(bev_encoder_path)

    parity_stress(
        bev_encoder_path,
        model.bev_encoder,
        bev_encoder_inputs,
        iters=args.iters,
        atol=args.atol,
        rtol=args.rtol,
        tag="bev_encoder",
    )

    # ----- 4) Aerial Image Encoder -----
    aer_encoder_path = f"{args.out_dir}/aer_encoder.onnx"
    aer_encoder_inputs = OrderedDict([("aer_img", torch.rand(1, 3, Haer, Waer, device="cuda", dtype=torch.float32))])
    aer_encoder = AerialImageEncoder(model.aer_backbone, model.aer_upernet, model.aer_head)
    aer_encoder = aer_encoder.cuda().eval()
    torch.onnx.export(
        aer_encoder,
        tuple(aer_encoder_inputs.values()),
        aer_encoder_path,
        opset_version=20,
        input_names=list(aer_encoder_inputs.keys()),
        output_names=["aer_feat"],
        dynamic_axes={},
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx.checker.check_model(aer_encoder_path)

    parity_stress(
        aer_encoder_path,
        aer_encoder,
        aer_encoder_inputs,
        iters=args.iters,
        atol=args.atol,
        rtol=args.rtol,
        tag="aer_encoder",
    )

    # ----- 5) Aerial Patch Sampler -----
    sampler_path = f"{args.out_dir}/aer_patch_sampler.onnx"
    sampler_inputs = OrderedDict(
        [
            ("aer_feat", torch.rand(1, OUT_DIM, Haer // Pa, Waer // Pa, dtype=torch.float32).cuda()),
            ("pose_uvr", torch.rand(1, N, 3, dtype=torch.float32).cuda()),
        ]
    )
    model.patch_sampler.return_mask = False

    torch.onnx.export(
        model.patch_sampler,
        tuple(sampler_inputs.values()),
        sampler_path,
        opset_version=20,
        input_names=list(sampler_inputs.keys()),
        output_names=["aer_patches_feat"],
        dynamic_axes={
            "pose_uvr": {1: "N"},  # [B, N, 3]
            "aer_patches_feat": {1: "N"},  # [B, N, C, Hb, Wb]
        },
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx.checker.check_model(sampler_path)

    parity_stress(
        sampler_path,
        model.patch_sampler,
        sampler_inputs,
        iters=args.iters,
        atol=args.atol,
        rtol=args.rtol,
        tag="aer_patch_sampler",
    )

    # ----- 6) BEV-Patch-PF Head -----
    head_path = f"{args.out_dir}/bev_patch_head.onnx"
    head_inputs = OrderedDict(
        [
            ("bev_feat", torch.rand(1, OUT_DIM + 2, Hbev, Wbev, dtype=torch.float32).cuda()),
            ("aer_patches_feat", torch.rand(1, N, OUT_DIM, Hbev, Wbev, dtype=torch.float32).cuda()),
            ("bev_mask", torch.rand(1, 1, Hbev, Wbev, dtype=torch.float32).cuda()),
        ]
    )
    model.head.return_only_score = True

    torch.onnx.export(
        model.head,
        tuple(head_inputs.values()),
        head_path,
        opset_version=20,
        input_names=list(head_inputs.keys()),
        output_names=["scores", "uncertainty"],
        dynamic_axes={
            "aer_patches_feat": {1: "N"},  # [B, N, C, Hb, Wb]
            "scores": {1: "N"},  # [B, N]
        },
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx.checker.check_model(head_path)

    parity_stress(
        head_path,
        model.head,
        head_inputs,
        iters=args.iters,
        atol=args.atol,
        rtol=args.rtol,
        tag="bev_patch_head",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bev_patch_net", help="model configuration file")
    parser.add_argument("--ckpt_path", type=str, required=True, help="model weights (.pth) path")
    parser.add_argument("--out_dir", type=str, required=True, help="ONNX output directory")
    parser.add_argument("--iters", default=10, type=int, help="# iters for stress testing")
    parser.add_argument("--atol", default=1e-3, type=float, help="abs tolerance for parity check")
    parser.add_argument("--rtol", default=1e-3, type=float, help="rel tolerance for parity check")
    args = parser.parse_args()

    args.model_config = f"config/model/{args.model}.yaml"

    os.makedirs(args.out_dir, exist_ok=True)
    export_onnx(args)
