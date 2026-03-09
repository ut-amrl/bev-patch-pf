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

H_GND, W_GND = 518, 518  # ground image resolution
H_AER, W_AER = 768, 768  # aerial image resolution
N_PARTICLES = 128  # number of hypotheses for aerial patch sampling


# ========== Wrappers for sub-modules ==========
class GroundEncoder(nn.Module):
    def __init__(self, gnd_backbone: nn.Module, gnd_upernet: nn.Module):
        super().__init__()
        self.gnd_backbone = gnd_backbone
        self.gnd_upernet = gnd_upernet

    @torch.inference_mode()
    def forward(self, gnd_img: torch.Tensor):
        return self.gnd_upernet(self.gnd_backbone(gnd_img))


class AerialEncoder(nn.Module):
    def __init__(self, aer_backbone: nn.Module, aer_decoder: nn.Module):
        super().__init__()
        self.aer_backbone = aer_backbone
        self.aer_decoder = aer_decoder

    @torch.inference_mode()
    def forward(self, aer_img: torch.Tensor):
        return self.aer_decoder(self.aer_backbone(aer_img))


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
    ort_outs = None
    pt_outs = None
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


def export_component_onnx(
    module: nn.Module,
    inputs: OrderedDict[str, torch.Tensor],
    out_dir: str,
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    tag: str,
    iters: int = 10,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    onnx_path = os.path.join(out_dir, f"{tag}.onnx")
    torch.onnx.export(
        module,
        tuple(inputs.values()),
        onnx_path,
        opset_version=20,
        input_names=list(inputs.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx.checker.check_model(onnx_path)
    parity_stress(onnx_path, module, inputs, iters=iters, atol=atol, rtol=rtol, tag=tag)


# ========== Main export function ==========
def export_onnx(args):
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = hydra.utils.instantiate(OmegaConf.load(args.model_config))
    load_checkpoint(args.ckpt_path, model)
    model = model.cuda().eval()

    Hbev, Wbev = model.bev_mapper.bev_shape

    # ----- 1) ground image encoder -----
    gnd_encoder_inputs = OrderedDict(
        [
            ("gnd_img", torch.rand(1, 3, H_GND, W_GND, dtype=torch.float32).cuda()),
        ]
    )
    gnd_encoder = GroundEncoder(model.gnd_backbone, model.gnd_upernet).cuda().eval()

    export_component_onnx(
        module=gnd_encoder,
        inputs=gnd_encoder_inputs,
        out_dir=args.out_dir,
        output_names=["gnd_emb"],
        dynamic_axes={},
        tag="gnd_encoder",
    )

    # ----- 2) BEV Mapper -----
    mapper_inputs = OrderedDict(
        [
            ("gnd_emb", torch.rand(1, model.gnd_hidden_dim, H_GND // 4, W_GND // 4, dtype=torch.float32).cuda()),
            ("depth", torch.rand(1, H_GND, W_GND, dtype=torch.float32).cuda()),
            ("K", torch.eye(3, dtype=torch.float32).unsqueeze(0).cuda()),
            ("cam2enu", torch.eye(4, dtype=torch.float32).unsqueeze(0).cuda()),
            ("resolution", torch.tensor(0.3, dtype=torch.float32).unsqueeze(0).cuda()),
        ]
    )

    export_component_onnx(
        module=model.bev_mapper,
        inputs=mapper_inputs,
        out_dir=args.out_dir,
        output_names=["bev_emb", "bev_mask"],
        dynamic_axes={"depth": {1: "H", 2: "W"}},
        tag="bev_mapper",
    )

    # ----- 3) BEV Encoder -----
    bev_encoder_inputs = OrderedDict(
        [
            ("bev_emb", torch.rand(1, model.gnd_hidden_dim, Hbev, Wbev, dtype=torch.float32).cuda()),
        ]
    )

    export_component_onnx(
        module=model.bev_encoder,
        inputs=bev_encoder_inputs,
        out_dir=args.out_dir,
        output_names=["bev_feat"],
        dynamic_axes={},
        tag="bev_encoder",
    )

    # ----- 4) Aerial Image Encoder -----
    aer_encoder_inputs = OrderedDict(
        [
            ("aer_img", torch.rand(1, 3, H_AER, W_AER, dtype=torch.float32).cuda()),
        ]
    )
    aer_encoder = AerialEncoder(model.aer_backbone, model.aer_decoder).cuda().eval()

    export_component_onnx(
        module=aer_encoder,
        inputs=aer_encoder_inputs,
        out_dir=args.out_dir,
        output_names=["aer_feat"],
        dynamic_axes={},
        tag="aer_encoder",
    )

    # ----- 5) Aerial Patch Sampler -----
    sampler_inputs = OrderedDict(
        [
            ("aer_feat", torch.rand(1, model.out_dim, H_AER // 4, W_AER // 4, dtype=torch.float32).cuda()),
            ("pose_uvr", torch.rand(1, N_PARTICLES, 3, dtype=torch.float32).cuda()),
        ]
    )

    export_component_onnx(
        module=model.patch_sampler,
        inputs=sampler_inputs,
        out_dir=args.out_dir,
        output_names=["aer_patches_feat"],
        dynamic_axes={
            "pose_uvr": {1: "N"},  # [B, N, 3]
            "aer_patches_feat": {1: "N"},  # [B, N, C, Hb, Wb]
        },
        tag="aer_patch_sampler",
    )

    # ----- 6) BEV-Patch-PF Head -----
    head_inputs = OrderedDict(
        [
            ("bev_feat", torch.rand(1, model.out_dim + 2, Hbev, Wbev, dtype=torch.float32).cuda()),
            ("aer_patches_feat", torch.rand(1, N_PARTICLES, model.out_dim, Hbev, Wbev, dtype=torch.float32).cuda()),
            ("bev_mask", torch.rand(1, 1, Hbev, Wbev, dtype=torch.float32).cuda()),
        ]
    )
    model.head.return_only_score = True

    export_component_onnx(
        module=model.head,
        inputs=head_inputs,
        out_dir=args.out_dir,
        output_names=["scores", "uncertainty"],
        dynamic_axes={
            "aer_patches_feat": {1: "N"},  # [B, N, C, Hb, Wb]
            "scores": {1: "N"},  # [B, N]
        },
        tag="bev_patch_head",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="config/model/bev_patch_net.yaml", help="path to model config YAML")
    parser.add_argument("--ckpt_path", type=str, required=True, help="model weights (.pth) path")
    parser.add_argument("--out_dir", type=str, required=True, help="ONNX output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    export_onnx(args)
