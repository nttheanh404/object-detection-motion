"""Post-processing helpers shared by grid demos and evaluation."""

from __future__ import annotations

import cv2
import numpy as np


def logits_to_probability(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))).astype(np.float32)


def connected_components(score: np.ndarray, threshold: float = 0.5, min_cells: int = 1) -> list[dict]:
    mask = (np.asarray(score) >= float(threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result: list[dict] = []
    for idx in range(1, count):
        x, y, w, h, area = map(int, stats[idx])
        if area < int(min_cells):
            continue
        values = score[y:y + h, x:x + w][labels[y:y + h, x:x + w] == idx]
        result.append({
            "x": x, "y": y, "w": w, "h": h, "cells": area,
            "score_mean": float(values.mean()) if values.size else 0.0,
            "score_max": float(values.max()) if values.size else 0.0,
        })
    return sorted(result, key=lambda item: item["score_max"], reverse=True)

