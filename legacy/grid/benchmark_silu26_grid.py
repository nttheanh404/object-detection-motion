#!/usr/bin/env python3
import argparse, sys, json, math
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import DataLoader
from train_silu26_grid import GridDataset, Silu26GridModel

def comps(mask):
    m=(mask.astype(np.uint8)*255)
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    areas=[int(st[i,cv2.CC_STAT_AREA]) for i in range(1,n)]
    return len(areas), areas

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--val',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/silu10_grid_train/val.tsv')
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--gt-th',type=float,default=0.25)
    ap.add_argument('--max-items',type=int,default=0)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    ck=torch.load(args.ckpt,map_location=device,weights_only=False)
    weights=ck.get('args',{}).get('weights','/home/ai5070/babyalpha/theanh/heatmap/yolov5_lumi.pt')
    model=Silu26GridModel(weights, freeze_backbone=True).to(device)
    model.load_state_dict(ck['model'],strict=False); model.eval()
    ds=GridDataset(args.val,max_items=args.max_items or None,augment=False)
    ld=DataLoader(ds,batch_size=args.batch,shuffle=False,num_workers=args.workers)
    ths=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    agg={t:{'tp':0,'fp':0,'fn':0,'pred_cells':0,'gt_cells':0,'neg_imgs':0,'neg_fp_imgs':0,'pos_imgs':0,'pos_hit_imgs':0,'components':[]} for t in ths}
    preview_dir=out/'preview'; preview_dir.mkdir(exist_ok=True)
    saved=0
    with torch.no_grad():
        for x,y,paths in ld:
            x=x.to(device); y=y.to(device)
            prob=torch.sigmoid(model(x)).cpu().numpy()[:,0]
            tgt=y.cpu().numpy()[:,0]
            gt=tgt>=args.gt_th
            for bi in range(prob.shape[0]):
                is_pos=gt[bi].any()
                for t in ths:
                    pred=prob[bi]>=t
                    tp=int((pred & gt[bi]).sum()); fp=int((pred & ~gt[bi]).sum()); fn=int((~pred & gt[bi]).sum())
                    a=agg[t]; a['tp']+=tp; a['fp']+=fp; a['fn']+=fn; a['pred_cells']+=int(pred.sum()); a['gt_cells']+=int(gt[bi].sum())
                    nc,areas=comps(pred); a['components'].append(nc)
                    if is_pos:
                        a['pos_imgs']+=1
                        if tp>0: a['pos_hit_imgs']+=1
                    else:
                        a['neg_imgs']+=1
                        if pred.any(): a['neg_fp_imgs']+=1
                if saved<16:
                    img=(x[bi].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
                    def over(m):
                        up=cv2.resize((np.clip(m,0,1)*255).astype(np.uint8),(576,352),interpolation=cv2.INTER_NEAREST)
                        color=cv2.applyColorMap(up,cv2.COLORMAP_TURBO)
                        return cv2.addWeighted(img,0.65,color,0.35,0)
                    sheet=np.concatenate([img,over(tgt[bi]),over(prob[bi])],axis=1)
                    cv2.putText(sheet,'image | target | pred',(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                    cv2.imwrite(str(preview_dir/f'bench_{saved:03d}.jpg'),sheet); saved+=1
    metrics=[]
    for t,a in agg.items():
        tp,fp,fn=a['tp'],a['fp'],a['fn']
        prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9); iou=tp/(tp+fp+fn+1e-9)
        metrics.append({'pred_th':t,'precision':prec,'recall':rec,'f1':f1,'iou':iou,'avg_pred_cells':a['pred_cells']/len(ds),'avg_gt_cells':a['gt_cells']/len(ds),'neg_fp_rate':a['neg_fp_imgs']/max(a['neg_imgs'],1),'pos_hit_rate':a['pos_hit_imgs']/max(a['pos_imgs'],1),'avg_components':float(np.mean(a['components']))})
    result={'ckpt':args.ckpt,'val':args.val,'num_items':len(ds),'gt_th':args.gt_th,'metrics':metrics}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with open(out/'metrics.md','w') as f:
        f.write('| pred_th | precision | recall | f1 | iou | neg_fp_rate | pos_hit_rate | avg_pred_cells | avg_components |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for m in metrics:
            f.write(f"| {m['pred_th']:.2f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['iou']:.3f} | {m['neg_fp_rate']:.3f} | {m['pos_hit_rate']:.3f} | {m['avg_pred_cells']:.1f} | {m['avg_components']:.2f} |\n")
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
