#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-60,60)))

def nms(boxes, scores, iou_th=0.45, topk=120):
    if len(boxes)==0: return []
    boxes=boxes.astype(np.float32); scores=scores.astype(np.float32)
    order=scores.argsort()[::-1][:topk]
    keep=[]
    while order.size:
        i=int(order[0]); keep.append(i)
        if order.size==1: break
        xx1=np.maximum(boxes[i,0], boxes[order[1:],0]); yy1=np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2=np.minimum(boxes[i,2], boxes[order[1:],2]); yy2=np.minimum(boxes[i,3], boxes[order[1:],3])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        ai=np.maximum(0,boxes[i,2]-boxes[i,0])*np.maximum(0,boxes[i,3]-boxes[i,1])
        ao=np.maximum(0,boxes[order[1:],2]-boxes[order[1:],0])*np.maximum(0,boxes[order[1:],3]-boxes[order[1:],1])
        iou=inter/np.maximum(ai+ao-inter,1e-6)
        order=order[1:][iou<=iou_th]
    return keep

def decode_output0(out0, conf_th=0.12):
    p=out0[0]
    xywh=p[:,:4]; obj=p[:,4]; cls=p[:,5:]
    if obj.max()>1 or obj.min()<0: obj=sigmoid(obj)
    if cls.max()>1 or cls.min()<0: cls=sigmoid(cls)
    # Lumi classes: person, head, pet. Chỉ lấy person.
    scores=obj*cls[:,0]
    m=scores>=conf_th
    if not np.any(m): return np.zeros((0,4),np.float32), np.zeros((0,),np.float32)
    xywh=xywh[m]; scores=scores[m]
    x,y,w,h=xywh[:,0],xywh[:,1],xywh[:,2],xywh[:,3]
    boxes=np.stack([x-w/2,y-h/2,x+w/2,y+h/2],axis=1).astype(np.float32)
    return boxes,scores.astype(np.float32)

def preprocess_lumi(frame_bgr):
    img=cv2.resize(frame_bgr,(576,352),interpolation=cv2.INTER_LINEAR)
    img=img[:,:,::-1].astype(np.float32)/255.0
    return np.transpose(img,(2,0,1))[None]

def heat_mask(mask,w,h):
    im=cv2.GaussianBlur(mask,(0,0),3)
    return cv2.applyColorMap(im,cv2.COLORMAP_TURBO)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--json',required=True)
    ap.add_argument('--max-frames',type=int,default=600)
    ap.add_argument('--out-w',type=int,default=960)
    ap.add_argument('--conf',type=float,default=0.10)
    ap.add_argument('--motion-th',type=float,default=0.32)
    ap.add_argument('--bbox-motion-ratio-th',type=float,default=0.045)
    ap.add_argument('--bbox-motion-score-th',type=float,default=0.18)
    args=ap.parse_args()

    sess=ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    inp=sess.get_inputs()[0].name
    cap=cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise RuntimeError(args.input)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    sw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow=args.out_w; oh=int(round(sh*ow/sw))
    writer=cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MJPG'), fps, (ow,oh))
    if not writer.isOpened(): raise RuntimeError(args.output)

    mog=cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=32,detectShadows=True)
    ko=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    persist=np.zeros((oh,ow),np.float32)
    sx=ow/576.0; sy=oh/352.0
    stats=[]
    for fi in range(args.max_frames):
        ok,frame=cap.read()
        if not ok: break
        t0=cv2.getTickCount()
        frame=cv2.resize(frame,(ow,oh),interpolation=cv2.INTER_AREA)

        # Motion robust, không model.
        blur=cv2.GaussianBlur(frame,(5,5),0)
        fg=mog.apply(blur)
        raw=(fg==255).astype(np.uint8)*255
        raw=cv2.morphologyEx(raw,cv2.MORPH_OPEN,ko)
        raw=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,kc)
        raw=cv2.dilate(raw,kd,iterations=1)
        persist=0.70*persist+0.30*(raw.astype(np.float32)/255.0)
        motion_score_img=np.clip(persist,0,1)
        motion_mask=(motion_score_img>args.motion_th).astype(np.uint8)*255

        # YOLO Lumi person detect.
        outs=sess.run(None,{inp:preprocess_lumi(frame)})
        boxes,scores=decode_output0(outs[0],args.conf)
        keep=nms(boxes,scores,0.45,120)
        boxes=boxes[keep] if len(keep) else np.zeros((0,4),np.float32)
        scores=scores[keep] if len(keep) else np.zeros((0,),np.float32)

        dets=[]
        final_mask=np.zeros((oh,ow),np.uint8)
        for b,conf in zip(boxes,scores):
            x1,y1,x2,y2=b
            x1=int(np.clip(x1*sx,0,ow-1)); x2=int(np.clip(x2*sx,0,ow-1))
            y1=int(np.clip(y1*sy,0,oh-1)); y2=int(np.clip(y2*sy,0,oh-1))
            if x2<=x1 or y2<=y1: continue
            roi_mask=motion_mask[y1:y2,x1:x2]
            roi_score=motion_score_img[y1:y2,x1:x2]
            area=(x2-x1)*(y2-y1)
            motion_pixels=int((roi_mask>0).sum())
            ratio=motion_pixels/max(area,1)
            # mean của top motion pixels ổn định hơn mean full bbox vì bbox người có nền trống.
            vals=roi_score.reshape(-1)
            if vals.size:
                topk=max(1,int(vals.size*0.15))
                top_mean=float(np.partition(vals,-topk)[-topk:].mean())
                mean_score=float(vals.mean())
            else:
                top_mean=0.0; mean_score=0.0
            moving = (ratio >= args.bbox_motion_ratio_th) or (top_mean >= args.bbox_motion_score_th)
            if moving:
                final_mask[y1:y2,x1:x2]=np.maximum(final_mask[y1:y2,x1:x2], roi_mask)
            dets.append({'box':[x1,y1,x2,y2],'conf':float(conf),'ratio':float(ratio),'top_motion_score':top_mean,'mean_motion_score':mean_score,'moving':bool(moving)})

        overlay=cv2.addWeighted(frame,0.84,heat_mask(final_mask,ow,oh),0.16,0)
        red=np.zeros_like(frame); red[:,:,2]=255
        overlay=np.where(final_mask[:,:,None]>0,cv2.addWeighted(overlay,0.60,red,0.40,0),overlay)
        moving_count=0
        for d in dets:
            x1,y1,x2,y2=d['box']
            if d['moving']:
                color=(0,255,0); thick=2; moving_count+=1; tag='M'
            else:
                color=(160,160,160); thick=1; tag='S'
            cv2.rectangle(overlay,(x1,y1),(x2,y2),color,thick)
            cv2.putText(overlay,f'{tag} {d["conf"]:.2f} r{d["ratio"]:.2f} t{d["top_motion_score"]:.2f}',(x1,max(13,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,0.36,color,1,cv2.LINE_AA)
        text=f'f {fi} yolo {len(dets)} moving_bbox {moving_count} motion_px {int((final_mask>0).sum())}'
        cv2.rectangle(overlay,(4,4),(ow-4,32),(0,0,0),-1)
        cv2.putText(overlay,text,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        writer.write(overlay)
        t1=cv2.getTickCount()
        stats.append({'frame':fi,'yolo_boxes':len(dets),'moving_boxes':moving_count,'time_ms':(t1-t0)*1000/cv2.getTickFrequency(),'dets':dets[:60]})

    writer.release(); cap.release()
    times=np.array([s['time_ms'] for s in stats],np.float32)
    summary={
        'model':args.model,'input':args.input,'output':args.output,'frames':len(stats),'src_size':[sw,sh],'out_size':[ow,oh],
        'conf':args.conf,'motion_th':args.motion_th,'bbox_motion_ratio_th':args.bbox_motion_ratio_th,'bbox_motion_score_th':args.bbox_motion_score_th,
        'algorithm':'YOLOv5 Lumi provides person bboxes; robust non-model motion decides whether each YOLO bbox is moving by motion overlap/top-score inside bbox.',
        'avg_time_ms':float(times.mean()) if len(times) else None,'p50_time_ms':float(np.percentile(times,50)) if len(times) else None,'p95_time_ms':float(np.percentile(times,95)) if len(times) else None,
        'avg_yolo_boxes':float(np.mean([s['yolo_boxes'] for s in stats])) if stats else None,'avg_moving_boxes':float(np.mean([s['moving_boxes'] for s in stats])) if stats else None,
        'stats':stats[:200]
    }
    Path(args.json).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='stats'},indent=2))
if __name__=='__main__': main()
