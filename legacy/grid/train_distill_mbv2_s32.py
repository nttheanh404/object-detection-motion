#!/usr/bin/env python3
import argparse, json, math, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

THIS=Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0,str(THIS))
from train_silu26_grid import GridDataset, Silu26GridModel, IMG_W, IMG_H, GRID_W, GRID_H, focal_bce

class MBV2S32Student(nn.Module):
    def __init__(self, pretrained=False, dec1=64, dec2=48, dec3=32):
        super().__init__()
        m=timm.create_model('mobilenetv2_050', pretrained=pretrained, num_classes=0, global_pool='')
        # Up to blocks.5: output [B,80,11,18] for 352x576. Cum params ~358,560.
        self.backbone=nn.Sequential(
            m.conv_stem, m.bn1,
            m.blocks[0], m.blocks[1], m.blocks[2], m.blocks[3], m.blocks[4], m.blocks[5]
        )
        self.decoder=nn.Sequential(
            nn.Conv2d(80, dec1, 1, bias=False),
            nn.BatchNorm2d(dec1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dec1, dec1, 3, padding=1, groups=dec1, bias=False),
            nn.BatchNorm2d(dec1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dec1, dec2, 1, bias=False),
            nn.BatchNorm2d(dec2),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(dec2, dec2, 3, padding=1, groups=dec2, bias=False),
            nn.BatchNorm2d(dec2),
            nn.SiLU(inplace=True),
            nn.Conv2d(dec2, dec3, 1, bias=False),
            nn.BatchNorm2d(dec3),
            nn.SiLU(inplace=True),
            nn.Conv2d(dec3, 1, 1, bias=True),
        )
    def forward(self,x):
        return self.decoder(self.backbone(x))

def count_params(m):
    return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def save_preview(student, teacher, ds, outdir, device, n=8):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    student.eval(); teacher.eval()
    for i in range(min(n,len(ds))):
        x,y,path=ds[i]
        xb=x[None].to(device)
        sp=torch.sigmoid(student(xb))[0,0].cpu().numpy()
        tp=torch.sigmoid(teacher(xb))[0,0].cpu().numpy()
        tgt=y[0].numpy()
        img=(x.numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
        def over(m,title):
            up=cv2.resize((np.clip(m,0,1)*255).astype(np.uint8),(IMG_W,IMG_H),interpolation=cv2.INTER_NEAREST)
            color=cv2.applyColorMap(up,cv2.COLORMAP_TURBO)
            out=cv2.addWeighted(img,0.65,color,0.35,0)
            cv2.rectangle(out,(0,0),(IMG_W,26),(0,0,0),-1)
            cv2.putText(out,title,(7,19),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
            return out
        sheet=np.hstack([img,over(tgt,'target'),over(tp,'teacher silu26'),over(sp,'student')])
        cv2.imwrite(str(outdir/f'preview_{i:03d}.jpg'),sheet)

@torch.no_grad()
def validate(student, teacher, loader, device, max_batches=80):
    student.eval(); teacher.eval()
    losses=[]; hard_losses=[]; dist_losses=[]; mse_losses=[]; pred_fracs=[]; teach_fracs=[]; tgt_fracs=[]
    for i,(x,y,_) in enumerate(loader):
        if max_batches and i>=max_batches: break
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        tlog=teacher(x); tprob=torch.sigmoid(tlog).detach()
        slog=student(x); sprob=torch.sigmoid(slog)
        hard=focal_bce(slog,y)
        dist=F.binary_cross_entropy_with_logits(slog,tprob)
        mse=F.mse_loss(sprob,tprob)
        loss=0.35*hard + 1.0*dist + 0.25*mse
        losses.append(float(loss.item())); hard_losses.append(float(hard.item())); dist_losses.append(float(dist.item())); mse_losses.append(float(mse.item()))
        pred_fracs.append(float((sprob>0.3).float().mean().item()))
        teach_fracs.append(float((tprob>0.3).float().mean().item()))
        tgt_fracs.append(float((y>0.25).float().mean().item()))
    return {
        'loss':float(np.mean(losses)) if losses else None,
        'hard_loss':float(np.mean(hard_losses)) if hard_losses else None,
        'distill_bce':float(np.mean(dist_losses)) if dist_losses else None,
        'distill_mse':float(np.mean(mse_losses)) if mse_losses else None,
        'target_pos_frac@0.25':float(np.mean(tgt_fracs)) if tgt_fracs else None,
        'teacher_pos_frac@0.3':float(np.mean(teach_fracs)) if teach_fracs else None,
        'student_pos_frac@0.3':float(np.mean(pred_fracs)) if pred_fracs else None,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default=str(THIS/'train.tsv'))
    ap.add_argument('--val',default=str(THIS/'val.tsv'))
    ap.add_argument('--teacher-ckpt',default=str(THIS/'run_silu26_finetune_from_silu10_v1/epoch_008.pt'))
    ap.add_argument('--lumi-weights',default='/home/ai5070/babyalpha/theanh/heatmap/yolov5_lumi.pt')
    ap.add_argument('--outdir',default=str(THIS/'run_distill_mbv2_s32_silu26_v1'))
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--target-gamma',type=float,default=1.0)
    ap.add_argument('--pretrained',action='store_true')
    ap.add_argument('--resume',default='')
    ap.add_argument('--quick-steps',type=int,default=0)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'

    teacher_ck=torch.load(args.teacher_ckpt,map_location=device,weights_only=False)
    ta=teacher_ck.get('args',{})
    teacher_weights=ta.get('weights',args.lumi_weights)
    teacher=Silu26GridModel(teacher_weights, freeze_backbone=bool(ta.get('freeze_backbone',False)), freeze_first=int(ta.get('freeze_first',0))).to(device)
    teacher.load_state_dict(teacher_ck['model'],strict=False)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad=False

    student=MBV2S32Student(pretrained=args.pretrained).to(device)
    print(f'device={device} student_params={count_params(student)} teacher_params={count_params(teacher)} output=44x72',flush=True)
    dummy=torch.zeros(1,3,IMG_H,IMG_W,device=device)
    with torch.no_grad():
        print('student_out',list(student(dummy).shape),'teacher_out',list(teacher(dummy).shape),flush=True)

    train_ds=GridDataset(args.train,augment=True,target_gamma=args.target_gamma)
    val_ds=GridDataset(args.val,augment=False,target_gamma=1.0)
    train_ld=DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True)
    val_ld=DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=args.workers,pin_memory=True)
    opt=torch.optim.AdamW(student.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))
    log=[]; step=0; start_epoch=0
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        student.load_state_dict(ck.get('student',ck.get('model',ck)),strict=False)
        if 'optimizer' in ck:
            try: opt.load_state_dict(ck['optimizer'])
            except Exception as e: print('optimizer resume skipped',e,flush=True)
        if 'scaler' in ck:
            try: scaler.load_state_dict(ck['scaler'])
            except Exception as e: print('scaler resume skipped',e,flush=True)
        log=ck.get('log',[]) or []
        step=int(ck.get('step',0) or (log[-1].get('step',0) if log else 0))
        start_epoch=int(ck.get('epoch',-1))+1 if 'epoch' in ck else len(log)
        print(f'resume start_epoch={start_epoch} step={step}',flush=True)

    for ep in range(start_epoch,args.epochs):
        student.train(); t0=time.time(); last={}
        for bi,(x,y,_) in enumerate(train_ld):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                tprob=torch.sigmoid(teacher(x)).detach()
            with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                slog=student(x); sprob=torch.sigmoid(slog)
                hard=focal_bce(slog,y)
                dist=F.binary_cross_entropy_with_logits(slog,tprob)
                mse=F.mse_loss(sprob,tprob)
                loss=0.35*hard + 1.0*dist + 0.25*mse
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step+=1
            last={'loss':float(loss.item()),'hard':float(hard.item()),'dist':float(dist.item()),'mse':float(mse.item())}
            if step%20==0:
                print(f'ep {ep} step {step} loss {last["loss"]:.5f} hard {last["hard"]:.5f} dist {last["dist"]:.5f} mse {last["mse"]:.5f}',flush=True)
            if args.quick_steps and step>=args.quick_steps: break
        val=validate(student,teacher,val_ld,device,max_batches=80)
        rec={'epoch':ep,'step':step,'train_last':last,'val':val,'time_sec':time.time()-t0}
        log.append(rec); print(json.dumps(rec),flush=True)
        ck={'student':student.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'args':vars(args),'epoch':ep,'step':step,'log':log,'student_params':count_params(student),'teacher_ckpt':args.teacher_ckpt,'grid':[GRID_W,GRID_H]}
        torch.save(ck,outdir/'last.pt'); torch.save(ck,outdir/f'epoch_{ep:03d}.pt')
        (outdir/'log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
        save_preview(student,teacher,val_ds,outdir/'preview',device,n=8)
        if args.quick_steps and step>=args.quick_steps: break
    student.eval(); dummy=torch.zeros(1,3,IMG_H,IMG_W,device=device)
    torch.onnx.export(student,dummy,str(outdir/'distill_mbv2_s32_grid.onnx'),input_names=['images'],output_names=['person_grid_logits'],opset_version=18)
    print('saved',outdir,flush=True)

if __name__=='__main__': main()
