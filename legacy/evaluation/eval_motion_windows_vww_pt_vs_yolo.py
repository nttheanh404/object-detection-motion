#!/usr/bin/env python3
from pathlib import Path
import argparse,json,cv2,numpy as np,torch,importlib.util
sweep_path=Path('/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/sweep_motion_window_grid_demo.py')
spec=importlib.util.spec_from_file_location('sw',sweep_path); sw=importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
conv_path=Path('/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/convert_vww_h5_to_pytorch.py')
spec2=importlib.util.spec_from_file_location('vwwconv',conv_path); vwwconv=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(vwwconv)

def containment(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); return inter/max(1e-6,(ax2-ax1)*(ay2-ay1))
def bbox_motion_stats(person, active, W,H,gw,gh):
    x1,y1,x2,y2=person; gx1=max(0,int(np.floor(x1*gw/W))); gy1=max(0,int(np.floor(y1*gh/H))); gx2=min(gw,int(np.ceil(x2*gw/W))); gy2=min(gh,int(np.ceil(y2*gh/H)))
    if gx2<=gx1 or gy2<=gy1: return 0,0,0.0
    roi=active[gy1:gy2,gx1:gx2]; ac=int(roi.sum()); return ac,int(roi.size),ac/max(1,int(roi.size))
def gbox_to_img(c,W,H,gw,gh): return [int(c['gx']*W/gw),int(c['gy']*H/gh),int((c['gx']+c['gw'])*W/gw),int((c['gy']+c['gh'])*H/gh)]
def prep(crop):
    img=cv2.resize(crop,(96,96),interpolation=cv2.INTER_AREA); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return torch.from_numpy(rgb.transpose(2,0,1)).float()
def draw(frame,active,crops,persons,args,label):
    H,W=frame.shape[:2]; out=frame.copy(); green=np.zeros_like(out); green[:,:,1]=255
    up=cv2.resize((active*255).astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST); out=np.where(up[:,:,None]>0,cv2.addWeighted(out,0.72,green,0.28,0),out); sw.draw_grid(out,args.grid_w,args.grid_h)
    for c,s,ok in crops:
        col=(0,255,255) if ok else (255,0,255); cv2.rectangle(out,(c[0],c[1]),(c[2],c[3]),col,2); cv2.putText(out,f'{s:.2f}',(c[0]+4,max(20,c[1]+20)),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,1,cv2.LINE_AA)
    for p in persons: cv2.rectangle(out,(p[0],p[1]),(p[2],p[3]),(0,255,0),2)
    cv2.rectangle(out,(0,0),(W,44),(0,0,0),-1); cv2.putText(out,label,(8,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--yolo-json',required=True); ap.add_argument('--ckpt',required=True); ap.add_argument('--outdir',required=True)
    ap.add_argument('--max-frames',type=int,default=300); ap.add_argument('--thresholds',default='0.3,0.5,0.7,0.8,0.9'); ap.add_argument('--grid-w',type=int,default=72); ap.add_argument('--grid-h',type=int,default=44); ap.add_argument('--win-w',type=int,default=12); ap.add_argument('--win-h',type=int,default=24); ap.add_argument('--stride',type=int,default=4); ap.add_argument('--cell-thr',type=float,default=0.30); ap.add_argument('--min-active',type=int,default=30); ap.add_argument('--nms-iou',type=float,default=0.45); ap.add_argument('--contain-thr',type=float,default=0.50); ap.add_argument('--eligible-motion-cells',type=int,default=2); ap.add_argument('--eligible-motion-ratio',type=float,default=0.02); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); thrs=[float(x) for x in args.thresholds.split(',')]
    device=args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    model=vwwconv.VWWMobileNetV1().to(device).eval(); ck=torch.load(args.ckpt,map_location=device,weights_only=False); model.load_state_dict(ck['model'])
    ydata=json.loads(Path(args.yolo_json).read_text()); yolo_frames={r['frame']:r['boxes'] for r in ydata['frames']}
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(args.input); fps=cap.get(cv2.CAP_PROP_FPS) or 25; W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); st=sw.make_motion_state()
    writers={thr:cv2.VideoWriter(str(outdir/f'pt_thr{thr:.2f}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)) for thr in thrs}
    stats={thr:[] for thr in thrs}
    for idx in range(args.max_frames):
        ok,frame=cap.read();
        if not ok: break
        score=sw.motion_score(frame,st); gs=cv2.resize(score,(args.grid_w,args.grid_h),interpolation=cv2.INTER_AREA); active=(gs>=args.cell_thr).astype(np.uint8)
        kept=sw.nms_windows(sw.scan_windows(active,gs,args.win_w,args.win_h,args.stride,args.min_active),args.nms_iou); raw_crops=[gbox_to_img(c,W,H,args.grid_w,args.grid_h) for c in kept]
        tensors=[]; valid=[]
        for c in raw_crops:
            crop=frame[c[1]:c[3],c[0]:c[2]]
            if crop.size: tensors.append(prep(crop)); valid.append(c)
        if tensors:
            with torch.no_grad(): probs=torch.softmax(model(torch.stack(tensors).to(device)),dim=1)[:,1].cpu().numpy().tolist()
        else: probs=[]
        persons=[b['xyxy'] for b in yolo_frames.get(idx,[])]
        eligible=[]
        for p in persons:
            mc,tc,mr=bbox_motion_stats(p,active,W,H,args.grid_w,args.grid_h); eligible.append(mc>=args.eligible_motion_cells and mr>=args.eligible_motion_ratio)
        for thr in thrs:
            passed=[c for c,s in zip(valid,probs) if s>=thr]
            all_hits=sum(1 for p in persons if max([containment(p,c) for c in passed],default=0)>=args.contain_thr)
            elig_total=sum(eligible); elig_hits=sum(1 for p,e in zip(persons,eligible) if e and max([containment(p,c) for c in passed],default=0)>=args.contain_thr)
            useful=sum(1 for c in passed if any(e and containment(p,c)>=args.contain_thr for p,e in zip(persons,eligible)))
            rec={'frame':idx,'yolo_persons':len(persons),'eligible':elig_total,'raw_crops':len(raw_crops),'passed_crops':len(passed),'all_hits':all_hits,'eligible_hits':elig_hits,'useful_passed':useful,'extra_passed':len(passed)-useful}
            stats[thr].append(rec)
            writers[thr].write(draw(frame,active,[(c,s,s>=thr) for c,s in zip(valid,probs)],persons,args,f'f{idx} PT>{thr:.2f} elig_hit={elig_hits}/{elig_total} raw={len(raw_crops)} pass={len(passed)}'))
    cap.release(); [w.release() for w in writers.values()]
    summary={}
    for thr,rows in stats.items():
        total_y=sum(r['yolo_persons'] for r in rows); hit=sum(r['all_hits'] for r in rows); elig=sum(r['eligible'] for r in rows); eh=sum(r['eligible_hits'] for r in rows)
        summary[str(thr)]={'frames':len(rows),'all_yolo_persons':total_y,'all_recall':hit/max(1,total_y),'eligible_yolo_persons':elig,'eligible_recall':eh/max(1,elig),'avg_raw_crops':float(np.mean([r['raw_crops'] for r in rows])),'avg_passed_crops':float(np.mean([r['passed_crops'] for r in rows])),'avg_useful_passed':float(np.mean([r['useful_passed'] for r in rows])),'avg_extra_passed':float(np.mean([r['extra_passed'] for r in rows])),'max_passed':int(max([r['passed_crops'] for r in rows],default=0))}
    Path(outdir/'summary.json').write_text(json.dumps({'args':vars(args),'summary':summary,'frames':stats},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
