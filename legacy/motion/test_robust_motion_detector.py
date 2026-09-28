#!/usr/bin/env python3
"""Small deterministic tests for the independent robust motion detector."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_motion_detector import RobustMotionDetector  # noqa: E402


def frame(value: int = 80, h: int = 180, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def run() -> None:
    # Static scene should settle to near zero.
    det = RobustMotionDetector(grid_width=32, grid_height=18)
    static = frame()
    first = det.process(static)
    assert np.max(first) == 0.0
    for _ in range(10):
        score = det.process(static)
    assert float(np.mean(score)) < 0.08, float(np.mean(score))

    # A rectangle crossing the centre must produce localized evidence.
    for x in range(20, 190, 20):
        moving = static.copy()
        cv2.rectangle(moving, (x, 65), (x + 55, 145), (220, 220, 220), -1)
        score = det.process(moving)
    centre = float(np.max(score[:, 8:24]))
    edge = float(np.max(np.concatenate((score[:, :4], score[:, -4:]), axis=1)))
    assert centre > edge + 0.03, (centre, edge)

    # Uniform exposure changes should not trigger a full-frame alarm.
    det.reset()
    det.process(frame(80))
    exposure = det.process(frame(150))
    assert float(np.mean(exposure)) < 0.15, float(np.mean(exposure))

    # A single bright speck is attenuated by the spatial component filter.
    det.reset()
    det.process(static)
    speck = static.copy()
    speck[90, 160] = (255, 255, 255)
    speck_score = det.process(speck)
    assert float(np.mean(speck_score)) < 0.12, float(np.mean(speck_score))

    print("robust_motion_detector tests: PASS")


if __name__ == "__main__":
    run()
