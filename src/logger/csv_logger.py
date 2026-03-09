import csv
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_1d_array(cell: str) -> np.ndarray:
    """Parses '[v1;v2;v3]' into a 1D numpy array."""
    if not cell or cell.strip() == "[]":
        return np.array([], dtype=float)
    values = cell.strip("[]").split(";")
    return np.array([float(v) for v in values], dtype=float)


def parse_2d_array(cell: str) -> np.ndarray:
    """Parses '[x1,y1,z1;x2,y2,z2]' into a 2D numpy array (N x 3)."""
    if not cell or cell.strip() == "[]":
        return np.empty((0, 0), dtype=float)

    rows = cell.strip("[]").split(";")
    return np.array([list(map(float, row.split(","))) for row in rows], dtype=float)


class CSVLogger:
    def __init__(self, log_path: str, columns: list[str]):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.columns = columns
        self.file = open(log_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns)

    def log(self, data: dict | list):
        if isinstance(data, dict):
            missing = set(self.columns) - set(data.keys())
            if missing:
                logger.warning(f"Missing columns in log: {missing}")
            row = [self._format_value(data.get(k, "")) for k in self.columns]
        elif isinstance(data, list):
            if len(data) != len(self.columns):
                logger.warning(
                    f"Data length {len(data)} does not match columns {len(self.columns)}"
                )
            row = [self._format_value(v) for v in data]
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.writer.writerow(row)
        self.file.flush()

    def _format_value(self, val):
        if isinstance(val, bool):
            return "True" if val else "False"
        if isinstance(val, (int, float)):
            return f"{val:.6f}" if isinstance(val, float) else str(val)
        elif isinstance(val, (list, tuple)):
            val = np.array(val)
        elif isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()

        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                return f"{val.item():.6f}"
            elif val.ndim == 1:
                return "[" + ";".join(f"{x:.6f}" for x in val) + "]"
            elif val.ndim == 2:
                return "[" + ";".join(",".join(f"{x:.6f}" for x in row) for row in val) + "]"
            else:
                raise ValueError(f"Unsupported array shape: {val.shape}")
        return str(val)

    def close(self):
        self.file.close()
        self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
