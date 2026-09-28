#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

ANCHORS = [
    np.array([[1.25,1.625],[2,3.75],[4.125,2.875]], np.float32) * 8,
    np.array([[1.875,3.8125],[3.875,2.8125],[3.6875,7.4375]], np.float32) * 16,
    np.array([[3.625,2.8125],[4.875,6.1875],[11.65625,10.1875]], np.float32) * 32,
]
STRIDES = [8,16,32]


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


def decode_output0(out0, conf_th=0.15):
    # output0 shape [1,N,8], assumed xywh + obj + 3 class probs/logits-ish. For Lumi: classes person, head, pet.
    p=out0[0]
    xywh=p[:,:4]
    obj=p[:,4]
    cls=p[:,5:]
    # detect whether already sigmoid/prob. Use sigmoid if values outside 0..1.
    if obj.max()>1 or obj.min()<0: obj=sigmoid(obj)
    if cls.max()>1 or cls.min()<0: cls=sigmoid(cls)
    person=cls[:,0]
    scores=obj*person
    m=scores>=conf_th
    if not np.any(m): return np.zeros((0,4),np.float32), np.zeros((0,),np.float32)
    xywh=xywh[m]; scores=scores[m]
    x,y,w,h=xywh[:,0],xywh[:,1],xywh[:,2],xywh[:,3]
    boxes=np.stack([x-w/2,y-h/2,x+w/2,y+h/2],axis=1)
    return boxes.astype(np.float32), scores.astype(np.float32)


def preprocess_lumi(frame_bgr):
    # Lumi input 576x352 RGB normalized 0..1 NCHW.
    img=cv2.resize(frame_bgr,(576,352),interpolation=cv2.INTER_LINEAR)
    img=img[:,:,::-1].astype(np.float32)/255.0
    return np.transpose(img,(2,0,1))[None]


def boxes_to_presence(boxes, scores, out_w, out_h, gw, gh, sx, sy):
    grid=np.zeros((gh,gw),np.float32)
    for b,s in zip(boxes,scores):
        x1,y1,x2,y2=b
        # Lumi box in input 576x352; map to output/demo frame.
        x1*=sx; x2*=sx; y1*=sy; y2*=sy
        # clamp
        x1=max(0,min(out_w-1,x1)); x2=max(0,min(out_w-1,x2)); y1=max(0,min(out_h-1,y1)); y2=max(0,min(out_h-1,y2))
        if x2<=x1 or y2<=y1: continue
        gx1=int(np.floor(x1/out_w*gw)); gx2=int(np.ceil(x2/out_w*gw))
        gy1=int(np.floor(y1/out_h*gh)); gy2=int(np.ceil(y2/out_h*gh))
        gx1=max(0,min(gw-1,gx1)); gx2=max(0,min(gw,gx2)); gy1=max(0,min(gh-1,gy1)); gy2=max(0,min(gh,gy2))
        pad=1
        grid[gy1:gy2,gx1:gx2]=np.maximum(grid[gy1:gy2,gx1:gx2],s)
        grid[max(0,gy1-pad):min(gh,gy2+pad),max(0,gx1-pad):min(gw,gx2+pad)]=np.maximum(grid[max(0,gy1-pad):min(gh,gy2+pad),max(0,gx1-pad):min(gw,gx2+pad)],s*0.45)
    return grid


def motion_grid_from_mask(mask, gw, gh):
    return cv2.resize(mask,(gw,gh),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0


def heat(grid,w,h):
    im=cv2.resize((np.clip(grid,0,1)*255).astype(np.uint8),(w,h),interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(im,cv2.COLORMAP_TURBO)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--json',required=True)
    ap.add_argument('--max-frames',type=int,default=600)
    ap.add_argument('--out-w',type=int,default=960)
    ap.add_argument('--grid-w',type=int,default=32)
    ap.add_argument('--grid-h',type=int,default=18)
    ap.add_argument('--conf',type=float,default=0.12)
    ap.add_argument('--motion-th',type=float,default=0.36)
    ap.add_argument('--soft-base',type=float,default=0.55)
    ap.add_argument('--final-th',type=float,default=0.26)
    args=ap.parse_args()
    sess=ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    inp=sess.get_inputs()[0].name
    cap=cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise RuntimeError(args.input)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    sw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow=args.out_w; oh=int(round(sh*ow/sw))
    writer=cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MJPG'), fps, (ow,oh))

    mog=cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=32,detectShadows=True)
    ko=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); kc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    persist=np.zeros((oh,ow),np.float32)
    presence_smooth=np.zeros((args.grid_h,args.grid_w),np.float32)
    stats=[]
    sx=ow/576.0; sy=oh/352.0
    for fi in range(args.max_frames):
        ok,frame=cap.read()
        if not ok: break
        t0=cv2.getTickCount()
        frame=cv2.resize(frame,(ow,oh),interpolation=cv2.INTER_AREA)
        blur=cv2.GaussianBlur(frame,(5,5),0)
        fg=mog.apply(blur)
        raw=(fg==255).astype(np.uint8)*255
        raw=cv2.morphologyEx(raw,cv2.MORPH_OPEN,ko)
        raw=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,kc)
        persist=0.70*persist+0.30*(raw.astype(np.float32)/255.0)
        motion_mask=(persist>args.motion_th).astype(np.uint8)*255
        mg=motion_grid_from_mask(motion_mask,args.grid_w,args.grid_h)

        yout=sess.run(None,{inp:preprocess_lumi(frame)})
        boxes,scores=decode_output0(yout[0],args.conf)
        keep=nms(boxes,scores,0.45,100)
        boxes=boxes[keep] if len(keep) else np.zeros((0,4),np.float32)
        scores=scores[keep] if len(keep) else np.zeros((0,),np.float32)
        pg=boxes_to_presence(boxes,scores,ow,oh,args.grid_w,args.grid_h,sx,sy)
        presence_smooth=0.70*presence_smooth+0.30*pg

        # Soft gate: YOLO không kill motion. Nó chỉ boost vùng có người.
        gate=args.soft_base+(1.0-args.soft_base)*np.clip(np.maximum(pg,presence_smooth),0,1)
        final=np.clip(mg*gate,0,1)
        mask_small=(final>=args.final_th).astype(np.uint8)*255
        fmask=cv2.resize(mask_small,(ow,oh),interpolation=cv2.INTER_NEAREST)
        n,lab,st,_=cv2.connectedComponentsWithStats(fmask,8)
        comps=[]
        for i in range(1,n):
            x,y,w,h,area=st[i]
            if area>=max(80,(ow*oh)//3000): comps.append((int(x),int(y),int(w),int(h),int(area)))

        overlay=cv2.addWeighted(frame,0.80,heat(final,ow,oh),0.20,0)
        red=np.zeros_like(frame); red[:,:,2]=255
        overlay=np.where(fmask[:,:,None]>0,cv2.addWeighted(overlay,0.58,red,0.42,0),overlay)
        # YOLO person prior boxes xanh lá, motion comps vàng.
        for b,s in zip(boxes,scores):
            if s<max(args.conf,0.18): continue
            x1,y1,x2,y2=b; x1=int(x1*sx); x2=int(x2*sx); y1=int(y1*sy); y2=int(y2*sy)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,220,0),1)
            cv2.putText(overlay,f'{s:.2f}',(x1,max(12,y1-2)),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,220,0),1,cv2.LINE_AA)
        for x,y,w,h,area in comps:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,255),2)
        text=f'f {fi} lumi_boxes {len(boxes)} motion_cells {(mg>=args.motion_th).sum()} final_cells {(final>=args.final_th).sum()} comps {len(comps)}'
        cv2.rectangle(overlay,(4,4),(ow-4,32),(0,0,0),-1)
        cv2.putText(overlay,text,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        writer.write(overlay)
        t1=cv2.getTickCount()
        stats.append({'frame':fi,'lumi_boxes':len(boxes),'final_cells':int((final>=args.final_th).sum()),'components':len(comps),'time_ms':(t1-t0)*1000/cv2.getTickFrequency()})
    writer.release(); cap.release()
    times=np.array([s['time_ms'] for s in stats],np.float32)
    summary={'model':args.model,'input':args.input,'output':args.output,'frames':len(stats),'src_size':[sw,sh],'out_size':[ow,oh],'grid':[args.grid_w,args.grid_h],'conf':args.conf,'motion_th':args.motion_th,'soft_base':args.soft_base,'final_th':args.final_th,'algorithm':'Robust MOG2 motion soft-gated by YOLOv5 Lumi person presence; final=motion*(base+(1-base)*presence), so YOLO does not kill motion','avg_time_ms':float(times.mean()) if len(times) else None,'p50_time_ms':float(np.percentile(times,50)) if len(times) else None,'p95_time_ms':float(np.percentile(times,95)) if len(times) else None,'avg_lumi_boxes':float(np.mean([s['lumi_boxes'] for s in stats])) if stats else None,'avg_components':float(np.mean([s['components'] for s in stats])) if stats else None,'stats':stats[:200]}
    Path(args.json).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='stats'},indent=2))
if __name__=='__main__': main()
