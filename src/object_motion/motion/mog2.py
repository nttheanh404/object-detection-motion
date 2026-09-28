"""Small stateful MOG2 baseline used for fixed-camera motion proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class MOG2Config:
    history: int = 500
    var_threshold: float = 36.0
    persist_alpha: float = 0.28
    dynamic_alpha: float = 0.003


class MOG2MotionDetector:
    """Return a soft per-pixel motion score in [0, 1]."""

    def __init__(self, config: MOG2Config | None = None) -> None:
        self.cfg = config or MOG2Config()
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=True,
        )
        self._mog.setShadowThreshold(0.55)
        self._mog.setBackgroundRatio(0.90)
        self._mog.setComplexityReductionThreshold(0.08)
        self._persist: np.ndarray | None = None
        self._dynamic: np.ndarray | None = None

    def reset(self) -> None:
        self.__init__(self.cfg)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty image")
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        fg = self._mog.apply(blurred, learningRate=-1)
        raw = (fg == 255).astype(np.uint8) * 255
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k_open)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_close)
        clean = cv2.dilate(clean, k_dilate, iterations=1).astype(np.float32) / 255.0
        if self._persist is None:
            self._persist = np.zeros_like(clean, dtype=np.float32)
            self._dynamic = np.zeros_like(clean, dtype=np.float32)
        assert self._dynamic is not None
        self._persist = (1.0 - self.cfg.persist_alpha) * self._persist + self.cfg.persist_alpha * clean
        self._dynamic = (1.0 - self.cfg.dynamic_alpha) * self._dynamic + self.cfg.dynamic_alpha * (self._persist > 0.20)
        suppression = np.clip(1.0 - 0.55 * np.maximum(self._dynamic - 0.45, 0.0) / 0.55, 0.45, 1.0)
        return np.clip(self._persist * suppression, 0.0, 1.0).astype(np.float32)

    def process_grid(self, frame_bgr: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
        score = self.process(frame_bgr)
        return cv2.resize(score, (int(grid_w), int(grid_h)), interpolation=cv2.INTER_AREA)

