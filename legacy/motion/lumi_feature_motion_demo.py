#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

def add_output(model_path,out_path,feature):
    m=onnx.load(model_path)
    names={o.name for o in m.graph.output}
    if feature not in names:
        m.graph.output.append(helper.make_tensor_value_info(feature,TensorProto.FLOAT,None))
    onnx.save(m,out_path)

def preprocess(frame):
    img=cv2.resize(frame,(576,352),interpolation=cv2.INTER_LINEAR)
    img=img[:,:,::-1].astype(np.float32)/255.0
    return np.transpose(img,(2,0,1))[None]

def feature_map(feat):
    a=feat[0].astype(np.float32)
    m=np.max(a,axis=0)
    lo=np.percentile(m,5); hi=np.percentile(m,99)
    if hi<=lo+1e-6: return np.zeros_like(m,np.float32)
    return np.clip((m-lo)/(hi-lo),0,1)

def heat_grid(grid,w,h):
    up=cv2.resize((np.clip(grid,0,1)*255).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(up,cv2.COLORMAP_TURBO),up

def draw_grid(img,gw,gh):
    h,w=img.shape[:2]
    for x in range(1,gw):
        xx=int(round(x*w/gw)); cv2.line(img,(xx,0),(xx,h),(45,45,45),1)
    for y in range(1,gh):
        yy=int(round(y*h/gh)); cv2.line(img,(0,yy),(w,yy),(45,45,45),1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='/home/ai5070/babyalpha/theanh/heatmap/yolov5_lumi.onnx')
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--json',required=True)
    ap.add_argument('--feature',default='silu_100')
    ap.add_argument('--max-frames',type=int,default=600)
    ap.add_argument('--out-w',type=int,default=960)
    ap.add_argument('--motion-th',type=float,default=0.32)
    ap.add_argument('--feature-percentile',type=float,default=80)
    ap.add_argument('--semantic-base',type=float,default=0.35)
    ap.add_argument('--final-th',type=float,default=0.18)
    ap.add_argument('--min-component-cells',type=int,default=2)
    ap.add_argument('--min-component-h-cells',type=int,default=1)
    ap.add_argument('--min-component-w-cells',type=int,default=1)
    ap.add_argument('--min-aspect-hw',type=float,default=0.35)
    ap.add_argument('--max-aspect-hw',type=float,default=8.0)
    args=ap.parse_args()

    out_model=Path(args.json).with_suffix(f'.{args.feature}.onnx')
    if not out_model.exists(): add_output(args.model,str(out_model),args.feature)
    sess=ort.InferenceSession(str(out_model),providers=['CPUExecutionProvider'])
    inp=sess.get_inputs()[0].name
    outputs=[o.name for o in sess.get_outputs()]
    feat_idx=outputs.index(args.feature)

    cap=cv2.VideoCapture(args.input)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    sw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); sh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ow=args.out_w; oh=int(round(sh*ow/sw))
    writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*'MJPG'),fps,(ow,oh))

    mog=cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=32,detectShadows=True)
    ko=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); kc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)); kd=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    persist=np.zeros((oh,ow),np.float32)
    semantic_smooth=None
    stats=[]
    map_shape=None
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
        raw=cv2.dilate(raw,kd,iterations=1)
        persist=0.70*persist+0.30*(raw.astype(np.float32)/255.0)
        motion_mask=(persist>args.motion_th).astype(np.uint8)*255

        outs=sess.run(None,{inp:preprocess(frame)})
        sem=feature_map(outs[feat_idx])
        gh,gw=sem.shape; map_shape=[int(gh),int(gw)]
        motion_grid=cv2.resize(motion_mask,(gw,gh),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        thr=np.percentile(sem,args.feature_percentile)
        sem_sparse=np.where((sem>=thr)&(sem>0.05),sem,0).astype(np.float32)
        if semantic_smooth is None:
            semantic_smooth=np.zeros_like(sem_sparse)
        semantic_smooth=0.65*semantic_smooth+0.35*sem_sparse
        semantic=np.maximum(sem_sparse,semantic_smooth)
        hard=motion_grid*(semantic>0).astype(np.float32)
        soft=motion_grid*(args.semantic_base+(1-args.semantic_base)*np.clip(semantic,0,1))
        final=np.maximum(hard,soft)
        final_mask_grid=(final>=args.final_th).astype(np.uint8)*255

        ngrid,glab,gst,_=cv2.connectedComponentsWithStats(final_mask_grid,8)
        filtered_grid=np.zeros_like(final_mask_grid)
        comps_grid=[]
        for gi in range(1,ngrid):
            gx,gy,gwc,ghc,garea=gst[gi]
            aspect=ghc/max(gwc,1)
            keep=(garea>=args.min_component_cells and ghc>=args.min_component_h_cells and gwc>=args.min_component_w_cells and aspect>=args.min_aspect_hw and aspect<=args.max_aspect_hw)
            if keep:
                filtered_grid[glab==gi]=255
                comps_grid.append((int(gx),int(gy),int(gwc),int(ghc),int(garea),float(aspect)))
        final_mask=cv2.resize(filtered_grid,(ow,oh),interpolation=cv2.INTER_NEAREST)
        n,lab,st,_=cv2.connectedComponentsWithStats(final_mask,8)
        comps=[]
        for i in range(1,n):
            x,y,w,h,area=st[i]
            comps.append((int(x),int(y),int(w),int(h),int(area)))

        color,up=heat_grid(final,ow,oh)
        overlay=cv2.addWeighted(frame,0.78,color,0.22,0)
        red=np.zeros_like(frame); red[:,:,2]=255
        overlay=np.where(final_mask[:,:,None]>0,cv2.addWeighted(overlay,0.58,red,0.42,0),overlay)
        draw_grid(overlay,sem.shape[1],sem.shape[0])
        for x,y,w,h,area in comps:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,255,255),2)
        text=f'f {fi} {args.feature}>{args.feature_percentile}% sem {(semantic>0).sum()} raw {(final>=args.final_th).sum()} kept {(filtered_grid>0).sum()} comps {len(comps)}'
        cv2.rectangle(overlay,(4,4),(ow-4,32),(0,0,0),-1)
        cv2.putText(overlay,text,(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        writer.write(overlay)
        t1=cv2.getTickCount()
        stats.append({'frame':fi,'semantic_cells':int((semantic>0).sum()),'final_cells':int((final>=args.final_th).sum()),'kept_cells':int((filtered_grid>0).sum()),'components':len(comps),'grid_components':len(comps_grid),'time_ms':(t1-t0)*1000/cv2.getTickFrequency()})
    writer.release(); cap.release()
    times=np.array([s['time_ms'] for s in stats],np.float32)
    summary={'model':args.model,'input':args.input,'output':args.output,'feature':args.feature,'map_shape':map_shape,'frames':len(stats),'src_size':[sw,sh],'out_size':[ow,oh],'motion_th':args.motion_th,'feature_percentile':args.feature_percentile,'semantic_base':args.semantic_base,'final_th':args.final_th,'min_component_cells':args.min_component_cells,'min_component_h_cells':args.min_component_h_cells,'min_aspect_hw':args.min_aspect_hw,'max_aspect_hw':args.max_aspect_hw,'algorithm':'Generic intermediate feature + robust motion grid + grid connected-component filter.','avg_time_ms':float(times.mean()) if len(times) else None,'p50_time_ms':float(np.percentile(times,50)) if len(times) else None,'p95_time_ms':float(np.percentile(times,95)) if len(times) else None,'avg_semantic_cells':float(np.mean([s['semantic_cells'] for s in stats])) if stats else None,'avg_final_cells':float(np.mean([s['final_cells'] for s in stats])) if stats else None,'avg_kept_cells':float(np.mean([s['kept_cells'] for s in stats])) if stats else None,'avg_components':float(np.mean([s['components'] for s in stats])) if stats else None,'stats':stats[:200]}
    Path(args.json).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='stats'},indent=2))
if __name__=='__main__': main()
