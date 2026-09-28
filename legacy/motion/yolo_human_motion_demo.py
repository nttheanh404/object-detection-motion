#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


def nms_xyxy(boxes, scores, iou_th=0.45, topk=80):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    order = scores.argsort()[::-1][:topk]
    keep = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        area_i = np.maximum(0, boxes[i,2]-boxes[i,0]) * np.maximum(0, boxes[i,3]-boxes[i,1])
        area_o = np.maximum(0, boxes[order[1:],2]-boxes[order[1:],0]) * np.maximum(0, boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter / np.maximum(area_i + area_o - inter, 1e-6)
        order = order[1:][iou <= iou_th]
    return keep


def yolo_person_candidates(sess, input_name, frame_bgr, conf_th=0.18, nms_th=0.45):
    h, w = frame_bgr.shape[:2]
    img = cv2.resize(frame_bgr, (320,320), interpolation=cv2.INTER_LINEAR)
    img = img[:,:,::-1].astype(np.float32) / 255.0
    inp = np.transpose(img, (2,0,1))[None]
    out = sess.run(None, {input_name: inp})[0]
    pred = out[0].T if out.shape[1] == 5 else out[0]
    scores = pred[:,4]
    m = scores >= conf_th
    pred = pred[m]; scores = scores[m]
    if len(pred) == 0:
        return []
    x,y,bw,bh = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
    boxes = np.stack([x-bw/2, y-bh/2, x+bw/2, y+bh/2], axis=1)
    boxes[:,[0,2]] *= w / 320.0
    boxes[:,[1,3]] *= h / 320.0
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, w-1)
    boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, h-1)
    keep = nms_xyxy(boxes, scores, nms_th, topk=120)
    return [(boxes[i], float(scores[i])) for i in keep]


def boxes_to_presence_grid(boxes, gw, gh, w, h):
    grid = np.zeros((gh, gw), np.float32)
    for b, s in boxes:
        x1,y1,x2,y2 = b
        gx1 = int(np.floor(x1 / w * gw)); gx2 = int(np.ceil(x2 / w * gw))
        gy1 = int(np.floor(y1 / h * gh)); gy2 = int(np.ceil(y2 / h * gh))
        gx1=max(0,min(gw-1,gx1)); gx2=max(0,min(gw,gx2))
        gy1=max(0,min(gh-1,gy1)); gy2=max(0,min(gh,gy2))
        if gx2 <= gx1: gx2 = min(gw, gx1+1)
        if gy2 <= gy1: gy2 = min(gh, gy1+1)
        grid[gy1:gy2, gx1:gx2] = np.maximum(grid[gy1:gy2, gx1:gx2], s)
        # mở rộng nhẹ vùng lân cận để presence không quá sắc theo bbox
        pad=1
        ax1=max(0,gx1-pad); ax2=min(gw,gx2+pad); ay1=max(0,gy1-pad); ay2=min(gh,gy2+pad)
        grid[ay1:ay2, ax1:ax2] = np.maximum(grid[ay1:ay2, ax1:ax2], s*0.45)
    return grid


def mask_to_grid(mask, gw, gh):
    return cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def grid_heat(grid, w, h, cmap=cv2.COLORMAP_TURBO):
    im = cv2.resize((np.clip(grid,0,1)*255).astype(np.uint8), (w,h), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(im, cmap)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--max-frames', type=int, default=400)
    ap.add_argument('--out-w', type=int, default=960)
    ap.add_argument('--grid-w', type=int, default=32)
    ap.add_argument('--grid-h', type=int, default=18)
    ap.add_argument('--conf', type=float, default=0.18)
    ap.add_argument('--human-motion-th', type=float, default=0.18)
    args=ap.parse_args()

    sess=ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name=sess.get_inputs()[0].name
    cap=cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise RuntimeError(args.input)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); src_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w=args.out_w; out_h=int(round(src_h*out_w/src_w))
    writer=cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MJPG'), fps, (out_w, out_h))
    if not writer.isOpened(): raise RuntimeError('writer')

    mog=cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
    k_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    k_close=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    motion_persist=np.zeros((out_h,out_w), np.float32)
    presence_smooth=np.zeros((args.grid_h,args.grid_w), np.float32)
    stats=[]

    for fi in range(args.max_frames):
        ok, frame=cap.read()
        if not ok: break
        t0=cv2.getTickCount()
        frame=cv2.resize(frame,(out_w,out_h),interpolation=cv2.INTER_AREA)

        # Classical motion, chỉ dùng để biết vùng nào đang thay đổi.
        blur=cv2.GaussianBlur(frame,(5,5),0)
        fg=mog.apply(blur)
        raw=(fg==255).astype(np.uint8)*255
        raw=cv2.morphologyEx(raw, cv2.MORPH_OPEN, k_open)
        raw=cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k_close)
        motion_persist=0.70*motion_persist+0.30*(raw.astype(np.float32)/255.0)
        motion_mask=(motion_persist>0.32).astype(np.uint8)*255
        motion_grid=mask_to_grid(motion_mask,args.grid_w,args.grid_h)

        # YOLO semantic prior: không coi đây là final detection, chỉ tạo human-presence grid.
        boxes=yolo_person_candidates(sess,input_name,frame,args.conf,0.45)
        presence=boxes_to_presence_grid(boxes,args.grid_w,args.grid_h,out_w,out_h)
        presence_smooth=0.65*presence_smooth+0.35*presence

        # Human-aware motion: mưa/cây/xe/background chỉ có motion nhưng thiếu person prior sẽ bị giảm.
        human_motion=np.clip(motion_grid * np.maximum(presence_smooth, presence),0,1)
        hm_mask_small=(human_motion>=args.human_motion_th).astype(np.uint8)*255
        hm_mask=cv2.resize(hm_mask_small,(out_w,out_h),interpolation=cv2.INTER_NEAREST)
        n,labels,stat,_=cv2.connectedComponentsWithStats(hm_mask,8)
        hm_boxes=[]
        for i in range(1,n):
            x,y,w,h,area=stat[i]
            if area>=max(80,(out_w*out_h)//2500): hm_boxes.append((int(x),int(y),int(w),int(h),int(area)))

        heat=grid_heat(human_motion,out_w,out_h)
        overlay=cv2.addWeighted(frame,0.78,heat,0.22,0)
        red=np.zeros_like(frame); red[:,:,2]=255
        overlay=np.where(hm_mask[:,:,None]>0,cv2.addWeighted(overlay,0.55,red,0.45,0),overlay)
        # YOLO relative boxes màu xanh lá mảnh, human-motion components màu vàng.
        for b,s in boxes:
            if s<0.28: continue
            x1,y1,x2,y2=map(int,b)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,220,0),1)
            cv2.putText(overlay,f'{s:.2f}',(x1,max(12,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,220,0),1,cv2.LINE_AA)
        for x,y,w,h,area in hm_boxes:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,255),2)

        text=f'f {fi} yolo_boxes {len(boxes)} human_motion_cells {(human_motion>=args.human_motion_th).sum()} hm_boxes {len(hm_boxes)}'
        cv2.rectangle(overlay,(4,4),(out_w-4,32),(0,0,0),-1)
        cv2.putText(overlay,text,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),1,cv2.LINE_AA)
        writer.write(overlay)
        t1=cv2.getTickCount()
        stats.append({'frame':fi,'yolo_boxes':len(boxes),'human_motion_cells':int((human_motion>=args.human_motion_th).sum()),'hm_boxes':len(hm_boxes),'time_ms':(t1-t0)*1000/cv2.getTickFrequency()})

    writer.release(); cap.release()
    times=np.array([s['time_ms'] for s in stats],np.float32)
    summary={
        'model':args.model,'input':args.input,'output':args.output,'frames':len(stats),'src_size':[src_w,src_h],'out_size':[out_w,out_h],
        'grid':[args.grid_w,args.grid_h],'conf':args.conf,'human_motion_th':args.human_motion_th,
        'algorithm':'YOLO person-presence grid * classical motion grid; YOLO boxes are relative semantic priors, not final detector/tracker',
        'avg_time_ms':float(times.mean()) if len(times) else None,'p50_time_ms':float(np.percentile(times,50)) if len(times) else None,'p95_time_ms':float(np.percentile(times,95)) if len(times) else None,
        'avg_yolo_boxes':float(np.mean([s['yolo_boxes'] for s in stats])) if stats else None,'avg_hm_boxes':float(np.mean([s['hm_boxes'] for s in stats])) if stats else None,
        'stats':stats[:200]
    }
    Path(args.json).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='stats'},indent=2))

if __name__=='__main__': main()
