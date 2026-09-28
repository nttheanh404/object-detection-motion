#!/usr/bin/env python3
"""Compare robust soft-grid motion against the MOG2 baseline on a video."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from object_motion import MOG2MotionDetector, RobustMotionDetector  # noqa: E402


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), q)) if values else 0.0


def render(frame: np.ndarray, grid: np.ndarray, threshold: float, title: str) -> np.ndarray:
    h, w = frame.shape[:2]
    heat = cv2.resize(np.clip(grid, 0, 1), (w, h), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    active = heat >= threshold
    out = cv2.addWeighted(frame, 0.58, color, 0.42, 0)
    muted = cv2.addWeighted(frame, 0.88, color, 0.12, 0)
    out[~active] = muted[~active]
    gh, gw = grid.shape
    for gx in range(1, gw):
        cv2.line(out, (int(gx * w / gw), 0), (int(gx * w / gw), h), (65, 65, 65), 1)
    for gy in range(1, gh):
        cv2.line(out, (0, int(gy * h / gh)), (w, int(gy * h / gh)), (65, 65, 65), 1)
    cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(out, f"{title} active={int((grid >= threshold).sum())}/{grid.size}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
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
        raise SystemExit(f"cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    robust = RobustMotionDetector(grid_width=args.grid_w, grid_height=args.grid_h)
    baseline = MOG2MotionDetector()
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(outdir / "robust_vs_mog2.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width * 2, height))
        if not writer.isOpened():
            raise SystemExit(f"cannot create output video in {outdir}")

    robust_ms: list[float] = []
    baseline_ms: list[float] = []
    robust_active: list[int] = []
    baseline_active: list[int] = []
    frame_count = 0
    robust_motion_frames = 0
    baseline_motion_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_count >= args.max_frames):
            break
        start = time.perf_counter()
        rgrid = robust.process(frame)
        robust_ms.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        bgrid = baseline.process_grid(frame, args.grid_w, args.grid_h)
        baseline_ms.append((time.perf_counter() - start) * 1000.0)
        ractive = int((rgrid >= args.threshold).sum())
        bactive = int((bgrid >= args.threshold).sum())
        robust_active.append(ractive)
        baseline_active.append(bactive)
        robust_motion_frames += int(ractive > 0)
        baseline_motion_frames += int(bactive > 0)
        if writer is not None:
            writer.write(np.concatenate((render(frame, rgrid, args.threshold, "ROBUST"),
                                         render(frame, bgrid, args.threshold, "MOG2")), axis=1))
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
            "p50_ms": percentile(robust_ms, 50), "p95_ms": percentile(robust_ms, 95),
            "mean_active_cells": float(np.mean(robust_active)) if robust_active else 0.0,
            "motion_frames": robust_motion_frames,
        },
        "mog2_baseline": {
            "mean_ms": float(np.mean(baseline_ms)) if baseline_ms else 0.0,
            "p50_ms": percentile(baseline_ms, 50), "p95_ms": percentile(baseline_ms, 95),
            "mean_active_cells": float(np.mean(baseline_active)) if baseline_active else 0.0,
            "motion_frames": baseline_motion_frames,
        },
        "note": "Motion accuracy requires frame-level motion labels; this command measures behavior and latency.",
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

