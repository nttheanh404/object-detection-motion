#!/usr/bin/env python3
from pathlib import Path
import argparse,json,cv2,numpy as np,torch,importlib.util,sys
conv_path=Path('/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/convert_vww_h5_to_pytorch.py')
spec=importlib.util.spec_from_file_location('vwwconv',conv_path); vwwconv=importlib.util.module_from_spec(spec); spec.loader.exec_module(vwwconv)
silu_dir=Path('/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/silu10_grid_train'); sys.path.insert(0,str(silu_dir))
from train_silu26_grid import Silu26GridModel

def read_rows(tsv,max_items=0):
    rows=[]
    for line in Path(tsv).read_text().splitlines():
        p=line.split('\t')
        if len(p)>=2: rows.append((p[0],p[1]))
    return rows[:max_items] if max_items else rows

def read_boxes(lab,W,H):
    boxes=[]; p=Path(lab)
    if not p.exists(): return boxes
    for line in p.read_text(errors='ignore').splitlines():
        ps=line.split()
        if len(ps)<5: continue
        try: cls=int(float(ps[0])); cx,cy,bw,bh=map(float,ps[1:5])
        except: continue
        if cls!=0: continue
        x1=(cx-bw/2)*W; y1=(cy-bh/2)*H; x2=(cx+bw/2)*W; y2=(cy+bh/2)*H
        x1=max(0,min(W,x1)); x2=max(0,min(W,x2)); y1=max(0,min(H,y1)); y2=max(0,min(H,y2))
        if x2>x1 and y2>y1: boxes.append([int(x1),int(y1),int(x2),int(y2)])
    return boxes

def containment(p,c):
    ix1=max(p[0],c[0]); iy1=max(p[1],c[1]); ix2=min(p[2],c[2]); iy2=min(p[3],c[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); return inter/max(1e-6,(p[2]-p[0])*(p[3]-p[1]))
def iou(a,b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3]); inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/max(1e-6,aa+bb-inter)
def nms(cands,thr=0.45):
    cands=sorted(cands,key=lambda x:x['score'],reverse=True); keep=[]
    for c in cands:
        if all(iou(c['box'],k['box'])<thr for k in keep): keep.append(c)
    return keep
def gen_windows(W,H,gw=72,gh=44,ww=12,wh=24,stride=4):
    out=[]
    for gy in range(0,gh-wh+1,stride):
        for gx in range(0,gw-ww+1,stride): out.append({'gx':gx,'gy':gy,'gw':ww,'gh':wh,'box':[int(gx*W/gw),int(gy*H/gh),int((gx+ww)*W/gw),int((gy+wh)*H/gh)]})
    return out
def prep_vww(img,box):
    c=img[box[1]:box[3],box[0]:box[2]]
    if c.size==0: c=np.zeros((96,96,3),np.uint8)
    c=cv2.resize(c,(96,96),interpolation=cv2.INTER_AREA); rgb=cv2.cvtColor(c,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return torch.from_numpy(rgb.transpose(2,0,1)).float()
def prep_silu(img):
    r=cv2.resize(img,(576,352),interpolation=cv2.INTER_LINEAR); rgb=cv2.cvtColor(r,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return torch.from_numpy(rgb.transpose(2,0,1))[None].float()
def sigmoid_np(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
def score_silu(prob,windows):
    out=[]
    for w in windows:
        roi=prob[w['gy']:w['gy']+w['gh'],w['gx']:w['gx']+w['gw']]
        flat=np.sort(roi.reshape(-1))[::-1]; k=min(20,len(flat)); s=float(flat[:k].mean())
        out.append({**w,'score':s})
    return out
def eval_scores(scores_by_img, boxes_by_img, thresholds, contain_thr=0.5):
    summary={}
    for th in thresholds:
        rows=[]
        for scored,boxes in zip(scores_by_img,boxes_by_img):
            passed=nms([c for c in scored if c['score']>=th],0.45)
            hits=sum(1 for b in boxes if max([containment(b,c['box']) for c in passed],default=0)>=contain_thr)
            useful=sum(1 for c in passed if any(containment(b,c['box'])>=contain_thr for b in boxes))
            rows.append({'boxes':len(boxes),'passed':len(passed),'hits':hits,'useful':useful,'extra':len(passed)-useful,'has_person':bool(boxes)})
        total=sum(r['boxes'] for r in rows); hit=sum(r['hits'] for r in rows); pos=sum(r['has_person'] for r in rows); poshit=sum(1 for r in rows if r['has_person'] and r['hits']>0); neg=sum(1 for r in rows if not r['has_person']); negfalse=sum(1 for r in rows if not r['has_person'] and r['passed']>0)
        summary[str(th)]={'frames':len(rows),'total_person_boxes':total,'box_recall':hit/max(1,total),'pos_frame_recall':poshit/max(1,pos),'neg_frame_false_rate':negfalse/max(1,neg),'avg_passed_windows':float(np.mean([r['passed'] for r in rows])),'avg_useful_windows':float(np.mean([r['useful'] for r in rows])),'avg_extra_windows':float(np.mean([r['extra'] for r in rows])),'max_passed':int(max([r['passed'] for r in rows],default=0))}
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--val',required=True); ap.add_argument('--vww-ckpt',required=True); ap.add_argument('--silu-ckpt',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--max-items',type=int,default=3000); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); device=args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    vww=vwwconv.VWWMobileNetV1().to(device).eval(); vww.load_state_dict(torch.load(args.vww_ckpt,map_location=device,weights_only=False)['model'])
    sc=torch.load(args.silu_ckpt,map_location=device,weights_only=False); weights=sc.get('args',{}).get('weights','/home/ai5070/babyalpha/theanh/heatmap/yolov5_lumi.pt'); silu=Silu26GridModel(weights,freeze_backbone=True).to(device).eval(); silu.load_state_dict(sc['model'],strict=False)
    rows=read_rows(args.val,args.max_items); boxes_by=[]; vww_scores=[]; silu_scores=[]
    for idx,(imgp,labp) in enumerate(rows):
        img=cv2.imread(imgp)
        if img is None: continue
        H,W=img.shape[:2]; boxes=read_boxes(labp,W,H); windows=gen_windows(W,H); boxes_by.append(boxes)
        with torch.no_grad():
            probs=torch.softmax(vww(torch.stack([prep_vww(img,w['box']) for w in windows]).to(device)),dim=1)[:,1].cpu().numpy()
            prob=sigmoid_np(silu(prep_silu(img).to(device))[0,0].detach().cpu().numpy())
        vww_scores.append([{**w,'score':float(s)} for w,s in zip(windows,probs)]); silu_scores.append(score_silu(prob,windows))
        if idx and idx%500==0: print('processed',idx,flush=True)
    vww_th=[0.3,0.5,0.7,0.8,0.9]; silu_th=[0.3,0.4,0.5,0.6,0.7]
    out={'args':vars(args),'windows_per_image':96,'vww':eval_scores(vww_scores,boxes_by,vww_th),'silu26':eval_scores(silu_scores,boxes_by,silu_th)}
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True); (outdir/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    md=['# No-motion window benchmark on validation labels: VWW vs Silu26','',f'Images: {len(boxes_by)}. Window 12×24 stride4: 96 windows/image.','', '## VWW fine-tuned','', '| thr | bbox recall | pos-frame recall | neg-frame false | avg pass | useful | extra | max |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for th,s in out['vww'].items(): md.append(f"| {th} | {s['box_recall']*100:.1f}% | {s['pos_frame_recall']*100:.1f}% | {s['neg_frame_false_rate']*100:.1f}% | {s['avg_passed_windows']:.2f} | {s['avg_useful_windows']:.2f} | {s['avg_extra_windows']:.2f} | {s['max_passed']} |")
    md += ['','## Silu26 window-score','', '| thr | bbox recall | pos-frame recall | neg-frame false | avg pass | useful | extra | max |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for th,s in out['silu26'].items(): md.append(f"| {th} | {s['box_recall']*100:.1f}% | {s['pos_frame_recall']*100:.1f}% | {s['neg_frame_false_rate']*100:.1f}% | {s['avg_passed_windows']:.2f} | {s['avg_useful_windows']:.2f} | {s['avg_extra_windows']:.2f} | {s['max_passed']} |")
    (outdir/'summary.md').write_text('\n'.join(md),encoding='utf-8'); print('\n'.join(md))
if __name__=='__main__': main()
