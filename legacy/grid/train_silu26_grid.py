#!/usr/bin/env python3
import argparse, sys, os, random, math, json, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

YOLOV5_SITE='/home/ai5070/babyalpha/miniconda3/envs/sam3_env/lib/python3.12/site-packages/yolov5'
if YOLOV5_SITE not in sys.path:
    sys.path.insert(0, YOLOV5_SITE)

IMG_W, IMG_H = 576, 352
GRID_W, GRID_H = 72, 44

class GridDataset(Dataset):
    def __init__(self, tsv, max_items=None, augment=False, target_gamma=1.0):
        rows=[]
        with open(tsv) as f:
            for line in f:
                p=line.rstrip('\n').split('\t')
                if len(p)>=2:
                    rows.append((p[0],p[1]))
        if max_items: rows=rows[:max_items]
        self.rows=rows
        self.augment=augment
        self.target_gamma=target_gamma
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        img_path, lab_path = self.rows[idx]
        img=cv2.imread(img_path)
        if img is None:
            # Return dummy negative if missing. Keeps training robust to stale manifest.
            img=np.zeros((IMG_H,IMG_W,3),np.uint8)
            target=np.zeros((GRID_H,GRID_W),np.float32)
            return torch.from_numpy(img.transpose(2,0,1)).float()/255.0, torch.from_numpy(target[None]), img_path
        img=cv2.resize(img,(IMG_W,IMG_H),interpolation=cv2.INTER_LINEAR)
        # simple horizontal flip
        flip = self.augment and random.random()<0.5
        if flip: img=cv2.flip(img,1)
        target=np.zeros((GRID_H,GRID_W),np.float32)
        try:
            lines=Path(lab_path).read_text(errors='ignore').strip().splitlines()
        except Exception:
            lines=[]
        for line in lines:
            p=line.split()
            if len(p)<5: continue
            try:
                cls=float(p[0]); cx=float(p[1]); cy=float(p[2]); bw=float(p[3]); bh=float(p[4])
            except: continue
            # For these datasets class 0 is person or person-only. Keep all nonempty as person prior for now.
            if flip: cx=1.0-cx
            x1=(cx-bw/2)*GRID_W; x2=(cx+bw/2)*GRID_W
            y1=(cy-bh/2)*GRID_H; y2=(cy+bh/2)*GRID_H
            x1=max(0.0,min(GRID_W,x1)); x2=max(0.0,min(GRID_W,x2))
            y1=max(0.0,min(GRID_H,y1)); y2=max(0.0,min(GRID_H,y2))
            if x2<=x1 or y2<=y1: continue
            ix1=max(0,int(math.floor(x1))); ix2=min(GRID_W,int(math.ceil(x2)))
            iy1=max(0,int(math.floor(y1))); iy2=min(GRID_H,int(math.ceil(y2)))
            # Linear cell overlap ratio = area(cell ∩ bbox) / area(cell), as discussed.
            for gy in range(iy1,iy2):
                yy1=max(y1,gy); yy2=min(y2,gy+1)
                if yy2<=yy1: continue
                for gx in range(ix1,ix2):
                    xx1=max(x1,gx); xx2=min(x2,gx+1)
                    if xx2<=xx1: continue
                    r=(xx2-xx1)*(yy2-yy1)
                    if r>target[gy,gx]: target[gy,gx]=r
        if self.target_gamma != 1.0:
            target=np.power(np.clip(target,0,1), self.target_gamma).astype(np.float32)
        img=img[:,:,::-1].copy() # RGB
        return torch.from_numpy(img.transpose(2,0,1)).float()/255.0, torch.from_numpy(target[None]), img_path

class Silu26GridModel(nn.Module):
    def __init__(self, weights, freeze_backbone=True, freeze_first=0):
        super().__init__()
        ckpt=torch.load(weights,map_location='cpu',weights_only=False)
        yolo=(ckpt.get('ema') or ckpt.get('model')).float().eval()
        self.backbone=nn.Sequential(*list(yolo.model[:5]))  # model.0..4 => silu_26 [B,256,44,72]
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad=False
        else:
            # Some YOLO checkpoints carry requires_grad=False. Re-enable first,
            # then optionally freeze the early modules.
            for p in self.backbone.parameters():
                p.requires_grad=True
            if freeze_first > 0:
                # Fine-tune only the later part of the Silu26 branch.
                # freeze_first=3 freezes model.0..2, i.e. up to Silu10.
                for m in list(self.backbone.children())[:freeze_first]:
                    for p in m.parameters():
                        p.requires_grad=False
        self.head=nn.Sequential(
            nn.Conv2d(256,96,3,padding=1,bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(96,1,1,bias=True),
        )
    def forward(self,x):
        f=self.backbone(x)
        return self.head(f)

def focal_bce(logits, target, alpha=0.75, gamma=2.0):
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    p=torch.sigmoid(logits)
    pt=torch.where(target>0.5,p,1-p)
    # soft targets: weight positives by target value but keep weak border cells useful
    aw=torch.where(target>0, alpha*torch.clamp(target,min=0.15), 1-alpha)
    loss=aw*((1-pt)**gamma)*bce
    return loss.mean()

@torch.no_grad()
def validate(model, loader, device, max_batches=50):
    model.eval(); losses=[]; pos=[]; pred=[]
    for i,(x,y,_) in enumerate(loader):
        if i>=max_batches: break
        x=x.to(device); y=y.to(device)
        o=model(x)
        losses.append(focal_bce(o,y).item())
        p=torch.sigmoid(o)
        pos.append((y>0).float().mean().item())
        pred.append((p>0.3).float().mean().item())
    return {'loss':float(np.mean(losses)) if losses else None,'target_pos_frac':float(np.mean(pos)) if pos else None,'pred_pos_frac@0.3':float(np.mean(pred)) if pred else None}

def save_preview(model, ds, outdir, device, n=8):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    model.eval()
    for i in range(min(n,len(ds))):
        x,y,path=ds[i]
        with torch.no_grad():
            p=torch.sigmoid(model(x[None].to(device)))[0,0].cpu().numpy()
        img=(x.numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
        tgt=y[0].numpy()
        def over(m):
            up=cv2.resize((np.clip(m,0,1)*255).astype(np.uint8),(IMG_W,IMG_H),interpolation=cv2.INTER_NEAREST)
            color=cv2.applyColorMap(up,cv2.COLORMAP_TURBO)
            return cv2.addWeighted(img,0.65,color,0.35,0)
        sheet=np.concatenate([img,over(tgt),over(p)],axis=1)
        cv2.putText(sheet,'image | target | pred',(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.imwrite(str(outdir/f'preview_{i:03d}.jpg'),sheet)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/silu10_grid_train/train.tsv')
    ap.add_argument('--val',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/silu10_grid_train/val.tsv')
    ap.add_argument('--weights',default='/home/ai5070/babyalpha/theanh/heatmap/yolov5_lumi.pt')
    ap.add_argument('--outdir',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/silu10_grid_train/run_silu26_v1')
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--max-train',type=int,default=0)
    ap.add_argument('--max-val',type=int,default=0)
    ap.add_argument('--unfreeze',action='store_true')
    ap.add_argument('--freeze-first',type=int,default=0,help='when --unfreeze is used, keep first N YOLO modules frozen; use 3 to freeze through Silu10')
    ap.add_argument('--quick-steps',type=int,default=0)
    ap.add_argument('--target-gamma',type=float,default=0.5,help='soft target exponent: overlap ** gamma; gamma<1 boosts border cells')
    ap.add_argument('--resume',default='',help='checkpoint path to resume/fine-tune from')
    ap.add_argument('--save-every-epoch',action='store_true',default=True)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    train_ds=GridDataset(args.train, max_items=args.max_train or None, augment=True, target_gamma=args.target_gamma)
    val_ds=GridDataset(args.val, max_items=args.max_val or None, augment=False, target_gamma=args.target_gamma)
    train_ld=DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True)
    val_ld=DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    model=Silu26GridModel(args.weights, freeze_backbone=not args.unfreeze, freeze_first=args.freeze_first).to(device)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    print(f'trainable_params={trainable} total_params={total} freeze_backbone={not args.unfreeze} freeze_first={args.freeze_first}', flush=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))
    log=[]; step=0; start_epoch=0
    if args.resume:
        ckpt=torch.load(args.resume,map_location=device,weights_only=False)
        state=ckpt.get('model', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'resume model from {args.resume} missing={len(missing)} unexpected={len(unexpected)}', flush=True)
        if 'optimizer' in ckpt:
            try:
                opt.load_state_dict(ckpt['optimizer'])
                print('resume optimizer ok', flush=True)
            except Exception as e:
                print(f'resume optimizer skipped: {e}', flush=True)
        if 'scaler' in ckpt:
            try:
                scaler.load_state_dict(ckpt['scaler'])
                print('resume scaler ok', flush=True)
            except Exception as e:
                print(f'resume scaler skipped: {e}', flush=True)
        log=ckpt.get('log', []) or []
        step=int(ckpt.get('step', 0) or (log[-1].get('step',0) if log else 0))
        start_epoch=int(ckpt.get('epoch', -1)) + 1 if 'epoch' in ckpt else len(log)
        print(f'resume start_epoch={start_epoch} step={step}', flush=True)
    for ep in range(start_epoch, args.epochs):
        model.train(); t0=time.time()
        for bi,(x,y,_) in enumerate(train_ld):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                o=model(x); loss=focal_bce(o,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            step+=1
            if step%20==0:
                print(f'ep {ep} step {step} loss {loss.item():.5f}',flush=True)
            if args.quick_steps and step>=args.quick_steps: break
        val=validate(model,val_ld,device,max_batches=50)
        rec={'epoch':ep,'step':step,'train_loss_last':float(loss.item()),'val':val,'time_sec':time.time()-t0}
        log.append(rec); print(json.dumps(rec),flush=True)
        ckpt={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'args':vars(args),'log':log,'epoch':ep,'step':step}
        torch.save(ckpt,outdir/'last.pt')
        if args.save_every_epoch:
            torch.save(ckpt,outdir/f'epoch_{ep:03d}.pt')
        save_preview(model,val_ds,outdir/'preview',device,n=8)
        if args.quick_steps and step>=args.quick_steps: break
    (outdir/'log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    # export onnx
    model.eval()
    dummy=torch.zeros(1,3,IMG_H,IMG_W,device=device)
    torch.onnx.export(model,dummy,str(outdir/'silu26_grid_student.onnx'),input_names=['images'],output_names=['person_grid_logits'],opset_version=18)
    print('saved',outdir)
if __name__=='__main__': main()
