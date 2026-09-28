#!/usr/bin/env python3
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
THIS=Path(__file__).resolve().parent
if str(THIS) not in sys.path: sys.path.insert(0,str(THIS))
from train_silu26_grid import GridDataset, IMG_W, IMG_H, GRID_W, GRID_H, focal_bce
from train_distill_mbv2_s32 import MBV2S32Student, count_params

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

@torch.no_grad()
def validate(student, loader, device, max_batches=80):
    student.eval(); vals=[]; hardv=[]; distv=[]; msev=[]; pred=[]; teach=[]; tgt=[]
    for i,(x,y,tprob,_) in enumerate(loader):
        if max_batches and i>=max_batches: break
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); tprob=tprob.to(device,non_blocking=True)
        slog=student(x); sprob=torch.sigmoid(slog)
        hard=focal_bce(slog,y); dist=F.binary_cross_entropy_with_logits(slog,tprob); mse=F.mse_loss(sprob,tprob)
        loss=0.35*hard+1.0*dist+0.25*mse
        vals.append(loss.item()); hardv.append(hard.item()); distv.append(dist.item()); msev.append(mse.item())
        pred.append((sprob>0.3).float().mean().item()); teach.append((tprob>0.3).float().mean().item()); tgt.append((y>0.25).float().mean().item())
    return {'loss':float(np.mean(vals)),'hard_loss':float(np.mean(hardv)),'distill_bce':float(np.mean(distv)),'distill_mse':float(np.mean(msev)),'target_pos_frac@0.25':float(np.mean(tgt)),'teacher_pos_frac@0.3':float(np.mean(teach)),'student_pos_frac@0.3':float(np.mean(pred))}

@torch.no_grad()
def save_preview(student, ds, outdir, device, n=8):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); student.eval()
    for i in range(min(n,len(ds))):
        x,y,t,path=ds[i]
        sp=torch.sigmoid(student(x[None].to(device)))[0,0].cpu().numpy(); tp=t[0].numpy(); tgt=y[0].numpy(); img=(x.numpy().transpose(1,2,0)[:,:,::-1]*255).astype(np.uint8)
        def over(m,title):
            up=cv2.resize((np.clip(m,0,1)*255).astype(np.uint8),(IMG_W,IMG_H),interpolation=cv2.INTER_NEAREST); color=cv2.applyColorMap(up,cv2.COLORMAP_TURBO); out=cv2.addWeighted(img,0.65,color,0.35,0)
            cv2.rectangle(out,(0,0),(IMG_W,26),(0,0,0),-1); cv2.putText(out,title,(7,19),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA); return out
        cv2.imwrite(str(outdir/f'preview_{i:03d}.jpg'),np.hstack([img,over(tgt,'target'),over(tp,'teacher cached'),over(sp,'student')]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default=str(THIS/'train.tsv'))
    ap.add_argument('--val',default=str(THIS/'val.tsv'))
    ap.add_argument('--train-cache',default=str(THIS/'teacher_cache_silu26_epoch8/train/teacher_prob_f16.npy'))
    ap.add_argument('--val-cache',default=str(THIS/'teacher_cache_silu26_epoch8/val/teacher_prob_f16.npy'))
    ap.add_argument('--outdir',default=str(THIS/'run_distill_mbv2_s32_silu26_v1'))
    ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--target-gamma',type=float,default=1.0)
    ap.add_argument('--resume',default='')
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    student=MBV2S32Student(pretrained=False).to(device)
    print(f'device={device} student_params={count_params(student)} output=44x72 cached_teacher=True',flush=True)
    train_ds=CachedTeacherDataset(args.train,args.train_cache,augment=True,target_gamma=args.target_gamma)
    val_ds=CachedTeacherDataset(args.val,args.val_cache,augment=False,target_gamma=1.0)
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
        log=ck.get('log',[]) or []; step=int(ck.get('step',0) or (log[-1].get('step',0) if log else 0)); start_epoch=int(ck.get('epoch',-1))+1 if 'epoch' in ck else len(log)
        print(f'resume start_epoch={start_epoch} step={step}',flush=True)
    for ep in range(start_epoch,args.epochs):
        student.train(); t0=time.time(); last={}
        for x,y,tprob,_ in train_ld:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); tprob=tprob.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                slog=student(x); sprob=torch.sigmoid(slog); hard=focal_bce(slog,y); dist=F.binary_cross_entropy_with_logits(slog,tprob); mse=F.mse_loss(sprob,tprob); loss=0.35*hard+1.0*dist+0.25*mse
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step+=1
            last={'loss':float(loss.item()),'hard':float(hard.item()),'dist':float(dist.item()),'mse':float(mse.item())}
            if step%50==0: print(f'ep {ep} step {step} loss {last["loss"]:.5f} hard {last["hard"]:.5f} dist {last["dist"]:.5f} mse {last["mse"]:.5f}',flush=True)
        val=validate(student,val_ld,device,max_batches=80); rec={'epoch':ep,'step':step,'train_last':last,'val':val,'time_sec':time.time()-t0}; log.append(rec); print(json.dumps(rec),flush=True)
        ck={'student':student.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'args':vars(args),'epoch':ep,'step':step,'log':log,'student_params':count_params(student),'grid':[GRID_W,GRID_H],'teacher_cache':{'train':args.train_cache,'val':args.val_cache}}
        torch.save(ck,outdir/'last_cached.pt'); torch.save(ck,outdir/f'epoch_{ep:03d}_cached.pt'); (outdir/'log_cached.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
        save_preview(student,val_ds,outdir/'preview_cached',device,n=8)
    torch.onnx.export(student,torch.zeros(1,3,IMG_H,IMG_W,device=device),str(outdir/'distill_mbv2_s32_grid_cached.onnx'),input_names=['images'],output_names=['person_grid_logits'],opset_version=18)
    print('saved',outdir,flush=True)
if __name__=='__main__': main()
