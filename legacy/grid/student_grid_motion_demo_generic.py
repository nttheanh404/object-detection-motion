#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

IMG_W, IMG_H = 576, 352
DEFAULT_GRID_W, DEFAULT_GRID_H = 144, 88


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def pct(a, q):
    if not a:
        return None
    return float(np.percentile(np.asarray(a, dtype=np.float32), q))


def connected_boxes_grid(score, thr, min_cells):
    mask = (score >= thr).astype(np.uint8)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if int(area) < min_cells:
            continue
        comp = labels[y:y+h, x:x+w] == i
        vals = score[y:y+h, x:x+w][comp]
        boxes.append({
            'gx': int(x), 'gy': int(y), 'gw': int(w), 'gh': int(h), 'cells': int(area),
            'score_mean': float(vals.mean()) if vals.size else 0.0,
            'score_max': float(vals.max()) if vals.size else 0.0,
        })
    boxes.sort(key=lambda b: b['score_max'], reverse=True)
    return boxes, mask


def grid_box_to_image(box, out_w, out_h):
    gw_grid = box.get('grid_w', DEFAULT_GRID_W)
    gh_grid = box.get('grid_h', DEFAULT_GRID_H)
    sx, sy = out_w / gw_grid, out_h / gh_grid
    x1 = int(round(box['gx'] * sx))
    y1 = int(round(box['gy'] * sy))
    x2 = int(round((box['gx'] + box['gw']) * sx))
    y2 = int(round((box['gy'] + box['gh']) * sy))
    x1 = max(0, min(out_w - 1, x1)); y1 = max(0, min(out_h - 1, y1))
    x2 = max(0, min(out_w - 1, x2)); y2 = max(0, min(out_h - 1, y2))
    return x1, y1, x2, y2


def make_motion_state(w, h, history, var_threshold):
    mog = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=True)
    mog.setShadowThreshold(0.55)
    mog.setBackgroundRatio(0.90)
    mog.setComplexityReductionThreshold(0.08)
    return {
        'mog': mog,
        'persist': np.zeros((h, w), np.float32),
        'dynamic_noise': np.zeros((h, w), np.float32),
        'k_open': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        'k_close': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        'k_dilate': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    }


def robust_motion(frame, st, persist_threshold):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    fg = st['mog'].apply(blur, learningRate=-1)
    raw = (fg == 255).astype(np.uint8) * 255
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, st['k_open'])
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, st['k_close'])
    clean = cv2.dilate(clean, st['k_dilate'], iterations=1)
    clean_f = clean.astype(np.float32) / 255.0
    st['persist'] = 0.72 * st['persist'] + 0.28 * clean_f
    st['dynamic_noise'] = 0.997 * st['dynamic_noise'] + 0.003 * (st['persist'] > 0.20).astype(np.float32)
    suppression = np.clip(1.0 - 0.55 * np.maximum(st['dynamic_noise'] - 0.45, 0.0) / 0.55, 0.45, 1.0)
    score_img = st['persist'] * suppression
    return score_img, (score_img >= persist_threshold).astype(np.uint8) * 255


def draw_overlay(frame, person_grid, motion_grid, combined, boxes, frame_idx, metrics, args):
    h, w = frame.shape[:2]
    heat = cv2.resize((np.clip(combined, 0, 1) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    out = cv2.addWeighted(frame, 0.72, heat_color, 0.28, 0)

    motion_up = cv2.resize((np.clip(motion_grid, 0, 1) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    green = np.zeros_like(frame); green[:, :, 1] = 255
    out = np.where(motion_up[:, :, None] > int(args.motion_draw_thr * 255), cv2.addWeighted(out, 0.75, green, 0.25, 0), out)

    for b in boxes:
        x1, y1, x2, y2 = grid_box_to_image(b, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{b['score_max']:.2f}/{b['cells']}"
        cv2.putText(out, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    text = (
        f"frame {frame_idx} boxes {len(boxes)} "
        f"read {metrics['read_ms']:.1f} motion {metrics['motion_ms']:.1f} "
        f"grid {metrics['grid_ms']:.1f} post {metrics['post_ms']:.1f} total {metrics['total_ms']:.1f} ms"
    )
    cv2.rectangle(out, (4, 4), (w - 4, 32), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/mot17_small/MOT17-02-FRCNN-raw.mp4')
    ap.add_argument('--onnx', default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/silu10_grid_train/run_full_v1/silu10_grid_student.onnx')
    ap.add_argument('--output', default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/mot17_small/MOT17-02_student_silu10_grid_motion_demo.mp4')
    ap.add_argument('--json', default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/mot17_small/MOT17-02_student_silu10_grid_motion_demo_summary.json')
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--start-frame', type=int, default=0)
    ap.add_argument('--history', type=int, default=500)
    ap.add_argument('--var-threshold', type=float, default=36.0)
    ap.add_argument('--motion-thr', type=float, default=0.42)
    ap.add_argument('--person-thr', type=float, default=0.50)
    ap.add_argument('--combined-thr', type=float, default=0.18)
    ap.add_argument('--min-cells', type=int, default=12)
    ap.add_argument('--motion-draw-thr', type=float, default=0.35)
    ap.add_argument('--providers', default='CPUExecutionProvider')
    args = ap.parse_args()

    providers = [p.strip() for p in args.providers.split(',') if p.strip()]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    inp_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {args.input}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), src_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'cannot open writer {out_path}')

    st = make_motion_state(w, h, args.history, args.var_threshold)
    frame_idx = args.start_frame
    stats = []
    processed = 0

    while processed < args.max_frames:
        t_frame0 = time.perf_counter()
        t0 = time.perf_counter()
        ok, frame = cap.read()
        read_ms = (time.perf_counter() - t0) * 1000.0
        if not ok:
            break

        t0 = time.perf_counter()
        motion_score, motion_mask = robust_motion(frame, st, args.motion_thr)
        motion_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        img = cv2.resize(frame, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(img, (2, 0, 1))[None]
        raw_out = sess.run(None, {inp_name: blob})[0]
        logits = raw_out[0, 0] if raw_out.ndim == 4 else raw_out.squeeze()
        person_grid = sigmoid(logits)
        grid_h, grid_w = person_grid.shape
        # Grid size follows model output, not hardcoded.
        motion_grid = cv2.resize(motion_score.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        grid_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        person_gate = np.clip((person_grid - args.person_thr) / max(1e-6, 1.0 - args.person_thr), 0, 1)
        combined = person_gate * np.clip(motion_grid, 0, 1)
        boxes, bin_mask = connected_boxes_grid(combined, args.combined_thr, args.min_cells)
        post_ms = (time.perf_counter() - t0) * 1000.0
        total_ms = (time.perf_counter() - t_frame0) * 1000.0

        metric = {
            'frame': int(frame_idx),
            'read_ms': float(read_ms),
            'motion_ms': float(motion_ms),
            'grid_ms': float(grid_ms),
            'post_ms': float(post_ms),
            'total_ms': float(total_ms),
            'person_grid_mean': float(person_grid.mean()),
            'person_grid_max': float(person_grid.max()),
            'motion_grid_mean': float(motion_grid.mean()),
            'motion_grid_max': float(motion_grid.max()),
            'combined_mean': float(combined.mean()),
            'combined_max': float(combined.max()),
            'boxes': [{**b, 'grid_w': grid_w, 'grid_h': grid_h, **dict(zip(['x1','y1','x2','y2'], grid_box_to_image({**b, 'grid_w': grid_w, 'grid_h': grid_h}, w, h)))} for b in boxes],
        }
        stats.append(metric)
        boxes_draw = [{**b, 'grid_w': grid_w, 'grid_h': grid_h} for b in boxes]
        writer.write(draw_overlay(frame, person_grid, motion_grid, combined, boxes_draw, frame_idx, metric, args))
        processed += 1
        frame_idx += 1

    writer.release(); cap.release()

    keys = ['read_ms', 'motion_ms', 'grid_ms', 'post_ms', 'total_ms']
    summary = {
        'input': args.input,
        'onnx': args.onnx,
        'output': str(out_path),
        'frames_processed': processed,
        'source_fps': float(src_fps),
        'width': w,
        'height': h,
        'model_input': [IMG_W, IMG_H],
        'grid': [grid_w if processed else None, grid_h if processed else None],
        'thresholds': {
            'motion_thr': args.motion_thr,
            'person_thr': args.person_thr,
            'combined_thr': args.combined_thr,
            'min_cells': args.min_cells,
        },
        'timing_ms': {
            k: {
                'mean': float(np.mean([s[k] for s in stats])) if stats else None,
                'p50': pct([s[k] for s in stats], 50),
                'p95': pct([s[k] for s in stats], 95),
            } for k in keys
        },
        'avg_boxes': float(np.mean([len(s['boxes']) for s in stats])) if stats else None,
        'frames_with_boxes': int(sum(1 for s in stats if s['boxes'])),
        'avg_person_grid_mean': float(np.mean([s['person_grid_mean'] for s in stats])) if stats else None,
        'avg_person_grid_max': float(np.mean([s['person_grid_max'] for s in stats])) if stats else None,
        'avg_combined_max': float(np.mean([s['combined_max'] for s in stats])) if stats else None,
        'frames': stats,
        'note': 'combined = clipped sigmoid(student_grid) gate multiplied by robust motion score; boxes are connected components on the model output grid scaled to video pixels.',
    }
    Path(args.json).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
