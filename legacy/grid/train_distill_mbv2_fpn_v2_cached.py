#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
THIS=Path(__file__).resolve().parent
if str(THIS) not in sys.path: sys.path.insert(0,str(THIS))
from train_silu26_grid import GridDataset, IMG_W, IMG_H, GRID_W, GRID_H, focal_bce

class CachedTeacherDataset(Dataset):
    def __init__(self, manifest, cache_npy, augment=False, target_gamma=1.0):
        self.base=GridDataset(manifest,augment=augment,target_gamma=target_gamma)
        self.cache=np.load(cache_npy,mmap_mode='r')
        assert len(self.base)==self.cache.shape[0], (len(self.base), self.cache.shape)
    def __len__(self): return len(self.base)
    def __getitem__(self,idx):
        x,y,path=self.base[idx]
        t=torch.from_numpy(np.array(self.cache[idx],dtype=np.float32))[None]
        return x,y,t,path

class DWBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,groups=ch,bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True),
            nn.Conv2d(ch,ch,1,bias=False), nn.BatchNorm2d(ch), nn.SiLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class MBV2FPNStudentV2(nn.Module):
    def __init__(self, pretrained=False, c8=24, c16=40, c32=64, fpn=48):
        super().__init__()
        m=timm.create_model('mobilenetv2_050',pretrained=pretrained,num_classes=0,global_pool='')
        self.stem=nn.Sequential(m.conv_stem,m.bn1,m.blocks[0],m.blocks[1])       # stride4
        self.b2=m.blocks[2]  # stride8, 16ch, 44x72
        self.b3=m.blocks[3]  # stride16, 32ch, 22x36
        self.b4=m.blocks[4]  # stride16, 48ch, 22x36
        self.b5=m.blocks[5]  # stride32, 80ch, 11x18
        self.lat8=nn.Sequential(nn.Conv2d(16,c8,1,bias=False),nn.BatchNorm2d(c8),nn.SiLU(inplace=True))
        self.lat16=nn.Sequential(nn.Conv2d(48,c16,1,bias=False),nn.BatchNorm2d(c16),nn.SiLU(inplace=True))
        self.lat32=nn.Sequential(nn.Conv2d(80,c32,1,bias=False),nn.BatchNorm2d(c32),nn.SiLU(inplace=True))
        self.merge16=nn.Sequential(nn.Conv2d(c16+c32,fpn,1,bias=False),nn.BatchNorm2d(fpn),nn.SiLU(inplace=True),DWBlock(fpn))
        self.merge8=nn.Sequential(nn.Conv2d(c8+fpn,fpn,1,bias=False),nn.BatchNorm2d(fpn),nn.SiLU(inplace=True),DWBlock(fpn),DWBlock(fpn))
        self.head=nn.Sequential(nn.Conv2d(fpn,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.SiLU(inplace=True),nn.Conv2d(32,1,1,bias=True))
    def forward(self,x):
        x=self.stem(x)
        f8=self.b2(x)
        f16=self.b4(self.b3(f8))
        f32=self.b5(f16)
        p32=self.lat32(f32)
        p16=torch.cat([self.lat16(f16),F.interpolate(p32,size=f16.shape[-2:],mode='nearest')],dim=1)
        p16=self.merge16(p16)
        p8=torch.cat([self.lat8(f8),F.interpolate(p16,size=f8.shape[-2:],mode='nearest')],dim=1)
        p8=self.merge8(p8)
        return self.head(p8)

def count_params(m): return sum(p.numel() for p in m.parameters())

def soft_dice_loss(prob,target,eps=1e-6):
    p=prob.flatten(1); t=target.flatten(1)
    inter=(p*t).sum(1); den=p.sum(1)+t.sum(1)
    return (1-(2*inter+eps)/(den+eps)).mean()

def boundary_map(x):
    # x [B,1,H,W], cheap soft gradient magnitude
    dx=torch.abs(x[:,:,:,1:]-x[:,:,:,:-1])
    dy=torch.abs(x[:,:,1:,:]-x[:,:,:-1,:])
    dx=F.pad(dx,(0,1,0,0)); dy=F.pad(dy,(0,0,0,1))
    return torch.clamp(dx+dy,0,1)

def loss_v2(logits, hard, teacher):
    prob=torch.sigmoid(logits)
    # weighted distill: emphasize teacher foreground and transition cells, reduce easy background dominance.
    trans=(teacher*(1-teacher))*4.0
    w=0.25 + 2.75*(teacher>0.3).float() + 1.25*trans
    wbce=F.binary_cross_entropy_with_logits(logits,teacher,reduction='none')
    wbce=(wbce*w).mean()
    dice=soft_dice_loss(prob,teacher)
    hardloss=focal_bce(logits,hard)
    pos_loss=(prob.mean(dim=(1,2,3))-teacher.mean(dim=(1,2,3))).abs().mean()
    edge_loss=F.l1_loss(boundary_map(prob),boundary_map(teacher))
    loss=0.20*hardloss + 1.00*wbce + 0.50*dice + 0.10*pos_loss + 0.10*edge_loss
    return loss, {'hard':hardloss,'wbce':wbce,'dice':dice,'pos':pos_loss,'edge':edge_loss}

@torch.no_grad()
def validate(model,loader,device,max_batches=80):
    model.eval(); acc={k:[] for k in ['loss','hard','wbce','dice','pos','edge','student_pos_frac@0.3','teacher_pos_frac@0.3','target_pos_frac@0.25']}
    for i,(x,y,t,_) in enumerate(loader):
        if max_batches and i>=max_batches: break
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); t=t.to(device,non_blocking=True)
        o=model(x); loss,parts=loss_v2(o,y,t); p=torch.sigmoid(o)
        acc['loss'].append(loss.item())
        for k,v in parts.items(): acc[k].append(v.item())
        acc['student_pos_frac@0.3'].append((p>0.3).float().mean().item()); acc['teacher_pos_frac@0.3'].append((t>0.3).float().mean().item()); acc['target_pos_frac@0.25'].append((y>0.25).float().mean().item())
    return {k:float(np.mean(v)) for k,v in acc.items()}

@torch.no_grad()
def save_preview(model,ds,outdir,device,n=8):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); model.eval()
    for i in range(min(n,len(ds))):
        x,y,t,path=ds[i]; sp=torch.sigmoid(model(x[None].to(device)))[0,0].cpu().numpy(); tp=t[0].numpy(); tgt=y[0].numpy(); img=(x.numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
        def over(m,title):
            up=cv2.resize((np.clip(m,0,1)*255).astype(np.uint8),(IMG_W,IMG_H),interpolation=cv2.INTER_NEAREST); color=cv2.applyColorMap(up,cv2.COLORMAP_TURBO); out=cv2.addWeighted(img,0.65,color,0.35,0)
            cv2.rectangle(out,(0,0),(IMG_W,26),(0,0,0),-1); cv2.putText(out,title,(7,19),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA); return out
        cv2.imwrite(str(outdir/f'preview_{i:03d}.jpg'),np.hstack([img,over(tgt,'hard'),over(tp,'teacher'),over(sp,'student-v2')]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default=str(THIS/'train.tsv')); ap.add_argument('--val',default=str(THIS/'val.tsv'))
    ap.add_argument('--train-cache',default=str(THIS/'teacher_cache_silu26_epoch8/train/teacher_prob_f16.npy'))
    ap.add_argument('--val-cache',default=str(THIS/'teacher_cache_silu26_epoch8/val/teacher_prob_f16.npy'))
    ap.add_argument('--outdir',default=str(THIS/'run_distill_mbv2_fpn_v2_silu26_v1'))
    ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch',type=int,default=64); ap.add_argument('--workers',type=int,default=0); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--target-gamma',type=float,default=1.0); ap.add_argument('--resume',default='')
    args=ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=MBV2FPNStudentV2(pretrained=False).to(device)
    print(f'device={device} params={count_params(model)} output={list(model(torch.zeros(1,3,IMG_H,IMG_W,device=device)).shape)}',flush=True)
    train_ds=CachedTeacherDataset(args.train,args.train_cache,augment=True,target_gamma=args.target_gamma); val_ds=CachedTeacherDataset(args.val,args.val_cache,augment=False,target_gamma=1.0)
    train_ld=DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True); val_ld=DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); scaler=torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))
    log=[]; step=0; start_epoch=0
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(ck.get('model',ck.get('student',ck)),strict=False)
        if 'optimizer' in ck:
            try: opt.load_state_dict(ck['optimizer'])
            except Exception as e: print('optimizer resume skipped',e,flush=True)
        if 'scaler' in ck:
            try: scaler.load_state_dict(ck['scaler'])
            except Exception as e: print('scaler resume skipped',e,flush=True)
        log=ck.get('log',[]) or []; step=int(ck.get('step',0)); start_epoch=int(ck.get('epoch',-1))+1 if 'epoch' in ck else len(log); print(f'resume start_epoch={start_epoch} step={step}',flush=True)
    for ep in range(start_epoch,args.epochs):
        model.train(); t0=time.time(); last={}
        for x,y,t,_ in train_ld:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); t=t.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                o=model(x); loss,parts=loss_v2(o,y,t)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step+=1
            last={'loss':float(loss.item()),**{k:float(v.item()) for k,v in parts.items()}}
            if step%50==0: print('ep',ep,'step',step,' '.join(f'{k} {v:.5f}' for k,v in last.items()),flush=True)
        val=validate(model,val_ld,device,max_batches=80); rec={'epoch':ep,'step':step,'train_last':last,'val':val,'time_sec':time.time()-t0}; log.append(rec); print(json.dumps(rec),flush=True)
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'args':vars(args),'epoch':ep,'step':step,'log':log,'params':count_params(model),'grid':[GRID_W,GRID_H]}
        torch.save(ck,outdir/'last.pt'); torch.save(ck,outdir/f'epoch_{ep:03d}.pt'); (outdir/'log.json').write_text(json.dumps(log,indent=2),encoding='utf-8'); save_preview(model,val_ds,outdir/'preview',device,n=8)
    torch.onnx.export(model,torch.zeros(1,3,IMG_H,IMG_W,device=device),str(outdir/'distill_mbv2_fpn_v2_grid.onnx'),input_names=['images'],output_names=['person_grid_logits'],opset_version=18); print('saved',outdir,flush=True)
if __name__=='__main__': main()
