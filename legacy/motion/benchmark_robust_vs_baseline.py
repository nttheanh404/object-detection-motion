#!/usr/bin/env python3
"""Benchmark the robust detector against the previous MOG2 baseline.

This benchmark measures detector behavior and latency.  It does not claim an
accuracy score without frame-level motion labels.  The output video is a
side-by-side visual diagnostic: robust soft-grid heatmap on the left and the
old binary MOG2 grid on the right.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_motion_detector import RobustMotionDetector  # noqa: E402


def percentile(values, p):
    return float(np.percentile(np.asarray(values, np.float64), p)) if values else 0.0


class MOG2Baseline:
    def __init__(self, grid_w, grid_h, work_w=320):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.work_w = work_w
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=True
        )
        self.mog.setShadowThreshold(0.55)
        self.mog.setBackgroundRatio(0.90)
        self.mog.setComplexityReductionThreshold(0.08)
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.persist = None

    def process(self, frame):
        h, w = frame.shape[:2]
        work_h = max(16, int(round(self.work_w * h / max(1, w))))
        small = cv2.resize(frame, (self.work_w, work_h), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        fg = self.mog.apply(small, learningRate=-1)
        raw = (fg == 255).astype(np.uint8) * 255
        clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, self.open_kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, self.close_kernel)
        clean = cv2.dilate(clean, self.dilate_kernel, iterations=1)
        score = clean.astype(np.float32) / 255.0
        if self.persist is None:
            self.persist = np.zeros_like(score, np.float32)
        self.persist = 0.72 * self.persist + 0.28 * score
        return cv2.resize(self.persist, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA)


def heat_overlay(frame, grid, threshold, title):
    h, w = frame.shape[:2]
    heat = cv2.resize(np.clip(grid, 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    active = heat >= threshold
    out = cv2.addWeighted(frame, 0.58, color, 0.42, 0)
    out[~active] = cv2.addWeighted(frame, 0.88, color, 0.12, 0)[~active]
    gh, gw = grid.shape
    for gx in range(1, gw):
        x = int(gx * w / gw)
        cv2.line(out, (x, 0), (x, h), (65, 65, 65), 1)
    for gy in range(1, gh):
        y = int(gy * h / gh)
        cv2.line(out, (0, y), (w, y), (65, 65, 65), 1)
    cv2.rectangle(out, (0, 0), (w, 38), (0, 0, 0), -1)
    active_cells = int(np.count_nonzero(grid >= threshold))
    cv2.putText(out, f"{title}  active={active_cells}/{grid.size}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--grid-w", type=int, default=32)
    ap.add_argument("--grid-h", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    robust = RobustMotionDetector(grid_width=args.grid_w, grid_height=args.grid_h)
    baseline = MOG2Baseline(args.grid_w, args.grid_h)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(outdir / "robust_vs_mog2.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height),
        )

    robust_ms, baseline_ms = [], []
    robust_active, baseline_active = [], []
    robust_frames = baseline_frames = 0
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_count >= args.max_frames:
            break
        t0 = time.perf_counter()
        rgrid = robust.process(frame)
        robust_ms.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        bgrid = baseline.process(frame)
        baseline_ms.append((time.perf_counter() - t0) * 1000.0)
        ractive = int(np.count_nonzero(rgrid >= args.threshold))
        bactive = int(np.count_nonzero(bgrid >= args.threshold))
        robust_active.append(ractive)
        baseline_active.append(bactive)
        robust_frames += ractive > 0
        baseline_frames += bactive > 0
        if writer is not None:
            left = heat_overlay(frame, rgrid, args.threshold, "ROBUST SOFT GRID")
            right = heat_overlay(frame, bgrid, args.threshold, "MOG2 BASELINE")
            writer.write(np.concatenate((left, right), axis=1))
        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()
    report = {
        "input": str(Path(args.input).resolve()),
        "frames": frame_count,
        "resolution": [width, height],
        "grid": [args.grid_h, args.grid_w],
        "threshold": args.threshold,
        "robust": {
            "mean_ms": float(np.mean(robust_ms)) if robust_ms else 0.0,
            "p50_ms": percentile(robust_ms, 50),
            "p95_ms": percentile(robust_ms, 95),
            "mean_active_cells": float(np.mean(robust_active)) if robust_active else 0.0,
            "motion_frames": robust_frames,
        },
        "mog2_baseline": {
            "mean_ms": float(np.mean(baseline_ms)) if baseline_ms else 0.0,
            "p50_ms": percentile(baseline_ms, 50),
            "p95_ms": percentile(baseline_ms, 95),
            "mean_active_cells": float(np.mean(baseline_active)) if baseline_active else 0.0,
            "motion_frames": baseline_frames,
        },
        "note": "This is a behavior/latency comparison. Accuracy requires frame-level motion labels.",
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
