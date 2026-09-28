"""Convert normalized YOLO person boxes into soft occupancy-grid targets."""

from __future__ import annotations

import math

import numpy as np


def yolo_boxes_to_soft_grid(
    boxes: list[tuple[float, float, float, float]],
    grid_w: int = 72,
    grid_h: int = 44,
    gamma: float = 1.0,
) -> np.ndarray:
    """Return cell occupancy from normalized (cx, cy, w, h) YOLO boxes.

    Each cell receives the area fraction covered by at least one box.  A
    gamma below one softly boosts border cells without making them binary.
    """
    target = np.zeros((int(grid_h), int(grid_w)), dtype=np.float32)
    for cx, cy, bw, bh in boxes:
        x1 = max(0.0, min(grid_w, (cx - bw / 2.0) * grid_w))
        x2 = max(0.0, min(grid_w, (cx + bw / 2.0) * grid_w))
        y1 = max(0.0, min(grid_h, (cy - bh / 2.0) * grid_h))
        y2 = max(0.0, min(grid_h, (cy + bh / 2.0) * grid_h))
        for gy in range(max(0, int(math.floor(y1))), min(grid_h, int(math.ceil(y2)))):
            for gx in range(max(0, int(math.floor(x1))), min(grid_w, int(math.ceil(x2)))):
                overlap = max(0.0, min(x2, gx + 1) - max(x1, gx)) * max(0.0, min(y2, gy + 1) - max(y1, gy))
                target[gy, gx] = max(target[gy, gx], float(overlap))
    if gamma != 1.0:
        target = np.power(np.clip(target, 0.0, 1.0), float(gamma)).astype(np.float32)
    return target

