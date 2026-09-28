#!/usr/bin/env python3
"""Stateful, low-compute motion detector with a soft spatial output.

This module is an independent implementation of a robust motion-detection
pipeline.  It is motivated by publicly documented requirements for camera
motion detection (soft grid output, tolerance to illumination/camera changes,
and rejection of small or repetitive noise), but it does not contain or claim
to reproduce any proprietary implementation.

The public interface is deliberately small:

    detector = RobustMotionDetector(grid_width=32, grid_height=24)
    score = detector.process(frame_bgr)  # float32 [grid_height, grid_width]
    detector.reset()

The returned grid is an amount-of-motion score in [0, 1], rather than a
thresholded mask.  Thresholding, zone filtering, and downstream object
classification remain outside this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionConfig:
    """Tuning parameters grouped by the part of the signal they affect."""

    work_width: int = 320
    work_height: Optional[int] = None
    history_size: int = 24
    # Per-pixel robust evidence thresholds.  They are expressed in normalized
    # luma units and are further adapted from the measured sensor noise.
    photo_floor: float = 0.020
    texture_floor: float = 0.012
    temporal_floor: float = 0.018
    # Fraction of a frame which may be strongly active before treating it as a
    # global event (illumination/camera instability), not local motion.
    global_fraction: float = 0.55
    # Maximum translation for which phase correlation is considered reliable.
    max_camera_shift_px: float = 12.0
    # Connected components below this fraction of the working image are treated
    # as isolated sensor/rain/dust transients.
    min_component_fraction: float = 0.00005
    # Background update rates.  Moving regions update very slowly, while stable
    # background updates quickly enough to follow gradual scene changes.
    background_alpha_stable: float = 0.025
    background_alpha_motion: float = 0.0015
    # Periodic suppression is deliberately conservative: repetitive motion is
    # discounted, not discarded, because a moving person can also be periodic.
    periodic_start: int = 8
    periodic_suppression: float = 0.35


class RobustMotionDetector:
    """Robust stateful detector returning a configurable soft motion grid.

    The detector is designed for a fixed camera.  It compensates global gain
    and offset changes, aligns small camera translations using phase
    correlation, combines photometric/texture/temporal evidence, removes tiny
    isolated components, and applies conservative temporal smoothing.
    """

    def __init__(
        self,
        grid_width: int = 32,
        grid_height: int = 24,
        config: Optional[MotionConfig] = None,
    ) -> None:
        if grid_width < 2 or grid_height < 2:
            raise ValueError("grid dimensions must be at least 2x2")
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.cfg = config or MotionConfig()
        self.reset()

    def reset(self) -> None:
        """Clear all temporal state without changing configuration."""

        self._prev: Optional[np.ndarray] = None
        self._background: Optional[np.ndarray] = None
        self._background_hp: Optional[np.ndarray] = None
        self._ema_grid = np.zeros((self.grid_height, self.grid_width), np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=self.cfg.history_size)
        self._frame_index = 0
        self._last_shift = (0.0, 0.0)
        self._last_phase_response = 0.0
        self._last_global_event = False

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def last_camera_shift(self) -> Tuple[float, float]:
        return self._last_shift

    @property
    def last_phase_response(self) -> float:
        return self._last_phase_response

    @property
    def last_global_event(self) -> bool:
        return self._last_global_event

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Process one BGR frame and return a float32 motion grid.

        The first frame initializes the state and returns an all-zero grid.  It
        does not require a user-provided motion-free warm-up sequence.
        """

        if frame_bgr is None or frame_bgr.ndim not in (2, 3):
            raise ValueError("frame_bgr must be a valid grayscale or BGR image")
        if frame_bgr.size == 0:
            raise ValueError("frame_bgr must not be empty")

        gray = self._prepare_luma(frame_bgr)
        hp = gray - cv2.GaussianBlur(gray, (0, 0), 4.0)

        if self._background is None:
            self._background = gray.copy()
            self._background_hp = hp.copy()
            self._prev = gray.copy()
            self._frame_index = 1
            return np.zeros((self.grid_height, self.grid_width), np.float32)

        assert self._prev is not None
        assert self._background_hp is not None

        aligned, aligned_hp, shift, response, camera_unstable = self._align_to_background(gray, hp)
        self._last_shift = shift
        self._last_phase_response = response

        # Correct a global brightness/exposure change before comparing pixels.
        # Percentiles are robust to a moving person occupying a small part of
        # the image and avoid a fragile all-pixel mean estimate.
        corrected, gain, offset = self._photometric_normalize(aligned, self._background)

        photo_residual = np.abs(corrected - self._background)
        texture_residual = np.abs(aligned_hp - self._background_hp)
        if camera_unstable:
            temporal_residual = np.zeros_like(photo_residual)
        else:
            temporal_residual = np.abs(corrected - self._prev)

        photo = self._soft_evidence(photo_residual, self.cfg.photo_floor)
        texture = self._soft_evidence(texture_residual, self.cfg.texture_floor)
        temporal = self._soft_evidence(temporal_residual, self.cfg.temporal_floor)

        # The texture channel is intentionally strong: uniform exposure
        # changes should not look like local object motion.
        pixel_score = 0.45 * photo + 0.35 * texture + 0.20 * temporal
        pixel_score = self._remove_isolated_components(pixel_score)

        strong = pixel_score >= 0.42
        active_fraction = float(np.mean(strong))
        spatial_std = float(np.std(photo_residual))
        global_event = (
            active_fraction >= self.cfg.global_fraction
            and spatial_std < 0.075
        )
        if camera_unstable:
            # A large camera change is not evidence that every cell contains a
            # moving object.  Keep a very small residual so recovery is smooth.
            pixel_score *= 0.08
        elif global_event:
            pixel_score *= 0.12
        self._last_global_event = bool(global_event or camera_unstable)

        current_grid = self._to_grid(pixel_score)
        current_grid = self._apply_periodic_suppression(current_grid)
        current_grid = np.clip(current_grid, 0.0, 1.0).astype(np.float32)

        # Fast attack/slow release keeps a short real event visible while
        # avoiding one-frame flicker from isolated noise.
        rising = current_grid >= self._ema_grid
        self._ema_grid = np.where(
            rising,
            0.55 * current_grid + 0.45 * self._ema_grid,
            0.12 * current_grid + 0.88 * self._ema_grid,
        ).astype(np.float32)

        self._update_background(corrected, aligned_hp, pixel_score, global_event, camera_unstable)
        self._prev = corrected.copy()
        self._frame_index += 1
        return np.clip(self._ema_grid, 0.0, 1.0).copy()

    def _prepare_luma(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            if frame.shape[2] == 1:
                frame = frame[:, :, 0]
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        work_h = self.cfg.work_height
        if work_h is None:
            work_h = max(16, int(round(self.cfg.work_width * h / max(1, w))))
        small = cv2.resize(
            frame,
            (int(self.cfg.work_width), int(work_h)),
            interpolation=cv2.INTER_AREA,
        )
        small = small.astype(np.float32) / 255.0
        return cv2.GaussianBlur(small, (3, 3), 0)

    def _align_to_background(
        self,
        gray: np.ndarray,
        hp: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], float, bool]:
        assert self._background is not None
        h, w = gray.shape
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        try:
            (dx, dy), response = cv2.phaseCorrelate(self._background, gray, window)
        except cv2.error:
            dx, dy, response = 0.0, 0.0, 0.0
        shift_norm = float(np.hypot(dx, dy))
        # Phase correlation is reliable for small translations.  For a large
        # jump, suppress this frame and let the state recover instead of
        # generating a full-frame alarm.
        reliable = response >= 0.08 and shift_norm <= self.cfg.max_camera_shift_px
        if not reliable or shift_norm < 0.75:
            return gray, hp, (float(dx), float(dy)), float(response), not reliable and shift_norm > 2.0
        matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        aligned = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        aligned_hp = cv2.warpAffine(hp, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return aligned, aligned_hp, (float(dx), float(dy)), float(response), False

    @staticmethod
    def _photometric_normalize(current: np.ndarray, background: np.ndarray) -> Tuple[np.ndarray, float, float]:
        c10, c90 = np.percentile(current, (10, 90))
        b10, b90 = np.percentile(background, (10, 90))
        gain = float((c90 - c10) / max(1e-4, b90 - b10))
        gain = float(np.clip(gain, 0.75, 1.35))
        offset = float(np.median(current - gain * background))
        corrected = np.clip((current - offset) / gain, 0.0, 1.0).astype(np.float32)
        return corrected, gain, offset

    @staticmethod
    def _soft_evidence(residual: np.ndarray, floor: float) -> np.ndarray:
        # Robustly estimate the current sensor/noise floor.  A MAD estimate is
        # stable even when a moving object occupies a meaningful region.
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        threshold = max(float(floor), med + 2.5 * mad)
        scale = max(0.018, 3.0 * mad, threshold * 0.75)
        z = (residual - threshold) / scale
        # Smoothly maps evidence to [0,1] without introducing a hard pixel
        # threshold that would make the grid unstable.
        return (1.0 / (1.0 + np.exp(-4.0 * z))).astype(np.float32)

    def _remove_isolated_components(self, score: np.ndarray) -> np.ndarray:
        strong = (score >= 0.48).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
        min_area = max(3, int(round(score.shape[0] * score.shape[1] * self.cfg.min_component_fraction)))
        keep = np.zeros(n, dtype=np.uint8)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                keep[i] = 1
        support = keep[labels].astype(np.float32)
        # Do not delete weak evidence before it has had a chance to accumulate;
        # only isolated strong specks are attenuated.
        return score * (0.35 + 0.65 * support)

    def _to_grid(self, pixel_score: np.ndarray) -> np.ndarray:
        # Mean preserves amount-of-motion semantics; a high percentile keeps a
        # small but real object from disappearing inside a large grid cell.
        gh, gw = self.grid_height, self.grid_width
        mean = cv2.resize(pixel_score, (gw, gh), interpolation=cv2.INTER_AREA)
        # A local max at grid resolution is too sensitive to single-pixel noise;
        # use a 3x3 blur as a spatial-support term instead.
        support = cv2.resize(cv2.GaussianBlur(pixel_score, (0, 0), 1.2), (gw, gh), interpolation=cv2.INTER_AREA)
        grid = 0.72 * mean + 0.28 * support
        return np.clip(grid, 0.0, 1.0).astype(np.float32)

    def _apply_periodic_suppression(self, current: np.ndarray) -> np.ndarray:
        self._history.append(current.copy())
        if len(self._history) < self.cfg.periodic_start:
            return current
        x = np.stack(self._history, axis=0).astype(np.float32)
        centered = x - np.mean(x, axis=0, keepdims=True)
        var = np.mean(centered * centered, axis=0)
        if x.shape[0] >= 5:
            a = centered[2:]
            b = centered[:-2]
            corr = np.sum(a * b, axis=0) / np.sqrt(
                np.sum(a * a, axis=0) * np.sum(b * b, axis=0) + 1e-6
            )
        else:
            corr = np.zeros_like(current)
        repetitive = np.clip((corr - 0.55) / 0.35, 0.0, 1.0)
        energetic = np.clip((var - 0.003) / 0.04, 0.0, 1.0)
        penalty = self.cfg.periodic_suppression * repetitive * energetic
        return current * (1.0 - penalty)

    def _update_background(
        self,
        corrected: np.ndarray,
        aligned_hp: np.ndarray,
        pixel_score: np.ndarray,
        global_event: bool,
        camera_unstable: bool,
    ) -> None:
        assert self._background is not None
        assert self._background_hp is not None
        if camera_unstable:
            alpha = np.full_like(pixel_score, 0.0002, dtype=np.float32)
        elif global_event:
            alpha = np.full_like(pixel_score, 0.004, dtype=np.float32)
        else:
            alpha = np.where(
                pixel_score < 0.30,
                self.cfg.background_alpha_stable,
                self.cfg.background_alpha_motion,
            ).astype(np.float32)
        self._background += alpha * (corrected - self._background)
        self._background_hp += alpha * (aligned_hp - self._background_hp)


__all__ = ["MotionConfig", "RobustMotionDetector"]

