#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
THIS=Path(__file__).resolve().parent
if str(THIS) not in sys.path: sys.path.insert(0,str(THIS))
from train_silu26_grid import Silu26GridModel, IMG_W, IMG_H, GRID_W, GRID_H
from train_distill_mbv2_s32 import MBV2S32Student
from train_distill_mbv2_fpn_v2_cached import MBV2FPNStudentV2

THRESHOLDS=[0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90]

def read_manifest(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        p=line.split('\t')
        if len(p)>=2: rows.append((p[0],p[1]))
    return rows

def parse_boxes(label_path):
    boxes=[]
    try: lines=Path(label_path).read_text(errors='ignore').splitlines()
    except Exception: lines=[]
    for ln in lines:
        vals=ln.split()
        if len(vals)<5: continue
        try:
            cls=int(float(vals[0])); cx,cy,w,h=map(float,vals[1:5])
        except Exception: continue
        # In these manifests class 0/person labels are what we care about.
        if cls!=0: continue
        x1=max(0.0,cx-w/2); y1=max(0.0,cy-h/2); x2=min(1.0,cx+w/2); y2=min(1.0,cy+h/2)
        if x2>x1 and y2>y1: boxes.append((x1,y1,x2,y2))
    return boxes

def shrink_box(b, frac=0.5):
    x1,y1,x2,y2=b; cx=(x1+x2)/2; cy=(y1+y2)/2; w=(x2-x1)*frac; h=(y2-y1)*frac
    return (max(0,cx-w/2),max(0,cy-h/2),min(1,cx+w/2),min(1,cy+h/2))

def cell_masks(boxes, gh=GRID_H, gw=GRID_W):
    # cell center inside bbox, not overlap area, because trigger seed is cell-center-like.
    yy=(np.arange(gh)+0.5)/gh; xx=(np.arange(gw)+0.5)/gw
    X,Y=np.meshgrid(xx,yy)
    full=np.zeros((gh,gw),bool); core=np.zeros((gh,gw),bool); per=[]
    for b in boxes:
        x1,y1,x2,y2=b
        m=(X>=x1)&(X<=x2)&(Y>=y1)&(Y<=y2)
        cb=shrink_box(b,0.5); cx1,cy1,cx2,cy2=cb
        cm=(X>=cx1)&(X<=cx2)&(Y>=cy1)&(Y<=cy2)
        # for very small person, core may be empty; fallback to full mask for hit metric
        if not cm.any(): cm=m.copy()
        full |= m; core |= cm; per.append({'full':m,'core':cm})
    return full, core, per

class ImgDataset(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        ip,lp=self.rows[idx]
        img=cv2.imread(ip)
        if img is None: img=np.zeros((IMG_H,IMG_W,3),np.uint8)
        img=cv2.resize(img,(IMG_W,IMG_H),interpolation=cv2.INTER_LINEAR)
        rgb=img[:,:,::-1].copy().astype(np.float32)/255.0
        boxes=parse_boxes(lp)
        return torch.from_numpy(rgb.transpose(2,0,1)), ip, lp, boxes

def collate(batch):
    xs=torch.stack([b[0] for b in batch],0)
    return xs, [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch]

def load_model(spec, device):
    typ=spec['type']; ckpt=spec['ckpt']
    if typ=='silu26':
        c=torch.load(ckpt,map_location='cpu',weights_only=False); args=c.get('args',{})
        m=Silu26GridModel(args.get('weights','/home/ai5070/babyalpha/theanh/heatmap/yolov5_lumi.pt'), freeze_backbone=False, freeze_first=int(args.get('freeze_first',0)))
        m.load_state_dict(c['model'],strict=False)
    elif typ=='mbv2_s32':
        c=torch.load(ckpt,map_location='cpu',weights_only=False); m=MBV2S32Student(pretrained=False); m.load_state_dict(c.get('student',c.get('model')),strict=False)
    elif typ=='mbv2_fpn_v2':
        c=torch.load(ckpt,map_location='cpu',weights_only=False); m=MBV2FPNStudentV2(pretrained=False); m.load_state_dict(c.get('model',c.get('student')),strict=False)
    else:
        raise ValueError(typ)
    return m.to(device).eval()

def init_acc():
    return {str(t):{
        'pred_cells':0,'tp_full_cells':0,'tp_core_cells':0,'fp_cells':0,'full_gt_cells':0,'core_gt_cells':0,'bg_cells':0,
        'persons':0,'person_hit1_full':0,'person_hit3_full':0,'person_hit1_core':0,'person_hit3_core':0,
        'pos_images':0,'pos_image_hit_full':0,'pos_image_hit_core':0,'neg_images':0,'neg_image_trigger':0,
        'frames':0,'frames_trigger':0
    } for t in THRESHOLDS}

def update(acc, prob, boxes):
    gh,gw=prob.shape
    full,core,per=cell_masks(boxes,gh,gw)
    bg=~full
    for t in THRESHOLDS:
        a=acc[str(t)]; pred=prob>=t
        pred_cells=int(pred.sum()); tp_full=int((pred&full).sum()); tp_core=int((pred&core).sum()); fp=int((pred&bg).sum())
        a['pred_cells']+=pred_cells; a['tp_full_cells']+=tp_full; a['tp_core_cells']+=tp_core; a['fp_cells']+=fp
        a['full_gt_cells']+=int(full.sum()); a['core_gt_cells']+=int(core.sum()); a['bg_cells']+=int(bg.sum())
        a['frames']+=1; a['frames_trigger']+=int(pred_cells>0)
        if boxes:
            a['pos_images']+=1; a['pos_image_hit_full']+=int(tp_full>0); a['pos_image_hit_core']+=int(tp_core>0)
            a['persons']+=len(per)
            for pm in per:
                hf=int((pred&pm['full']).sum()); hc=int((pred&pm['core']).sum())
                a['person_hit1_full']+=int(hf>=1); a['person_hit3_full']+=int(hf>=3)
                a['person_hit1_core']+=int(hc>=1); a['person_hit3_core']+=int(hc>=3)
        else:
            a['neg_images']+=1; a['neg_image_trigger']+=int(pred_cells>0)

def finalize(acc):
    rows=[]
    for t in THRESHOLDS:
        a=acc[str(t)]
        pred=max(1,a['pred_cells']); persons=max(1,a['persons']); pos=max(1,a['pos_images']); neg=max(1,a['neg_images']); frames=max(1,a['frames'])
        rows.append({
            'threshold':t,
            'cell_precision_full':a['tp_full_cells']/pred if a['pred_cells'] else None,
            'cell_precision_core':a['tp_core_cells']/pred if a['pred_cells'] else None,
            'cell_recall_full':a['tp_full_cells']/a['full_gt_cells'] if a['full_gt_cells'] else None,
            'cell_recall_core':a['tp_core_cells']/a['core_gt_cells'] if a['core_gt_cells'] else None,
            'fp_cells_per_frame':a['fp_cells']/frames,
            'pred_cells_per_frame':a['pred_cells']/frames,
            'trigger_rate_all':a['frames_trigger']/frames,
            'person_hit1_full':a['person_hit1_full']/persons,
            'person_hit3_full':a['person_hit3_full']/persons,
            'person_hit1_core':a['person_hit1_core']/persons,
            'person_hit3_core':a['person_hit3_core']/persons,
            'pos_image_recall_full':a['pos_image_hit_full']/pos,
            'pos_image_recall_core':a['pos_image_hit_core']/pos,
            'neg_image_false_trigger':a['neg_image_trigger']/neg if a['neg_images'] else None,
            'raw_counts':a,
        })
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--val',default=str(THIS/'val.tsv')); ap.add_argument('--outdir',default=str(THIS/'grid_detection_metrics_recent')); ap.add_argument('--batch',type=int,default=64); ap.add_argument('--workers',type=int,default=0); ap.add_argument('--max-items',type=int,default=0)
    args=ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    rows=read_manifest(args.val); 
    if args.max_items: rows=rows[:args.max_items]
    ds=ImgDataset(rows); ld=DataLoader(ds,batch_size=args.batch,shuffle=False,num_workers=args.workers,collate_fn=collate)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    specs=[
        {'name':'silu26_teacher_epoch8','type':'silu26','ckpt':str(THIS/'run_silu26_finetune_from_silu10_v1/epoch_008.pt')},
        {'name':'student_mbv2_s32_epoch19','type':'mbv2_s32','ckpt':str(THIS/'run_distill_mbv2_s32_silu26_v1/epoch_019_cached.pt')},
        {'name':'student_mbv2_fpn_v2_epoch2','type':'mbv2_fpn_v2','ckpt':str(THIS/'run_distill_mbv2_fpn_v2_silu26_v1/last.pt')},
    ]
    allres=[]
    for spec in specs:
        if not Path(spec['ckpt']).exists():
            allres.append({'name':spec['name'],'missing':spec['ckpt']}); continue
        t0=time.time(); model=load_model(spec,device); acc=init_acc(); n=0
        with torch.no_grad():
            for x,ips,lps,boxes_batch in ld:
                x=x.to(device,non_blocking=True).float(); prob=torch.sigmoid(model(x)).detach().cpu().numpy()[:,0]
                for p,boxes in zip(prob,boxes_batch): update(acc,p,boxes); n+=1
        res={'name':spec['name'],'type':spec['type'],'ckpt':spec['ckpt'],'device':device,'items':n,'time_sec':time.time()-t0,'metrics':finalize(acc)}
        allres.append(res); print(spec['name'], 'done', n, 'time', round(res['time_sec'],1), flush=True)
    (outdir/'metrics.json').write_text(json.dumps(allres,indent=2),encoding='utf-8')
    # markdown summary: selected thresholds
    lines=['# Grid detection metrics theo bbox/person\n\n','Val set: `%s`, items: %d. Grid positive = `score >= threshold`. GT cell = tâm cell nằm trong bbox person. Core = bbox shrink 50%%.\n\n'%(args.val,len(rows))]
    lines.append('| Model | th | person hit@1 full | person hit@3 full | image recall | cell precision | FP cells/frame | pred cells/frame | neg false trigger |\n')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in allres:
        if 'metrics' not in r: continue
        for th in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]:
            m=next(x for x in r['metrics'] if abs(x['threshold']-th)<1e-9)
            def pct(v): return 'n/a' if v is None else f'{100*v:.1f}%'
            lines.append(f"| {r['name']} | {th:.2f} | {pct(m['person_hit1_full'])} | {pct(m['person_hit3_full'])} | {pct(m['pos_image_recall_full'])} | {pct(m['cell_precision_full'])} | {m['fp_cells_per_frame']:.1f} | {m['pred_cells_per_frame']:.1f} | {pct(m['neg_image_false_trigger'])} |\n")
    lines.append('\nGhi chú: threshold tốt cho filter phải ưu tiên `person hit@1/full` và `image recall` rất cao, rồi mới xem `FP cells/frame` và false trigger.\n')
    (outdir/'metrics.md').write_text(''.join(lines),encoding='utf-8')
    print(outdir/'metrics.md')
if __name__=='__main__': main()
