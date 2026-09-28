#!/usr/bin/env python3
from pathlib import Path
import argparse, json, time, random, importlib.util
import cv2, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

conv_path=Path('/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/convert_vww_h5_to_pytorch.py')
spec=importlib.util.spec_from_file_location('vwwconv',conv_path)
vwwconv=importlib.util.module_from_spec(spec); spec.loader.exec_module(vwwconv)

class CropDataset(Dataset):
    def __init__(self, manifest, img_size=96, train=False):
        self.rows=[]; self.img_size=img_size; self.train=train
        for line in Path(manifest).read_text().splitlines():
            p=line.split('\t')
            if len(p)>=2: self.rows.append((p[0],int(p[1])))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        path,label=self.rows[idx]
        img=cv2.imread(path)
        if img is None: img=np.zeros((self.img_size,self.img_size,3),np.uint8)
        else: img=cv2.resize(img,(self.img_size,self.img_size),interpolation=cv2.INTER_AREA)
        if self.train:
            if random.random()<0.5: img=cv2.flip(img,1)
            if random.random()<0.25:
                img=np.clip(img.astype(np.float32)*random.uniform(0.95,1.05)+random.uniform(-5,5),0,255).astype(np.uint8)
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        return torch.from_numpy(rgb.transpose(2,0,1)).float(), torch.tensor(label,dtype=torch.long)

def class_counts(ds):
    c=[0,0]
    for _,y in ds.rows: c[y]+=1
    return c

def evaluate(model, loader, device, thresholds=(0.3,0.5,0.7,0.8,0.9)):
    model.eval(); ce=nn.CrossEntropyLoss(); losses=[]; ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            logits=model(x); losses.append(float(ce(logits,y).item()))
            prob=torch.softmax(logits,dim=1)[:,1]
            ys.append(y.cpu().numpy()); ps.append(prob.cpu().numpy())
    y=np.concatenate(ys) if ys else np.zeros((0,),np.int64); p=np.concatenate(ps) if ps else np.zeros((0,),np.float32)
    out={'loss':float(np.mean(losses)) if losses else None,'n':int(len(y)),'pos_frac':float(y.mean()) if len(y) else 0.0}
    for th in thresholds:
        pred=p>=th
        tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
        out[f'th{th}']={'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'pred_pos_frac':float(pred.mean()) if len(pred) else 0.0}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True); ap.add_argument('--val',required=True); ap.add_argument('--init',required=True); ap.add_argument('--outdir',required=True)
    ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--img-size',type=int,default=96)
    ap.add_argument('--freeze-backbone-epochs',type=int,default=1); ap.add_argument('--resume',default='')
    args=ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    train_ds=CropDataset(args.train,args.img_size,train=True); val_ds=CropDataset(args.val,args.img_size,train=False)
    counts=class_counts(train_ds); val_counts=class_counts(val_ds)
    weights=[1.0/max(1,counts[y]) for _,y in train_ds.rows]
    sampler=WeightedRandomSampler(weights,num_samples=len(weights),replacement=True)
    train_loader=DataLoader(train_ds,batch_size=args.batch,sampler=sampler,num_workers=args.workers,pin_memory=device.startswith('cuda'))
    val_loader=DataLoader(val_ds,batch_size=args.batch*2,shuffle=False,num_workers=args.workers,pin_memory=device.startswith('cuda'))
    model=vwwconv.VWWMobileNetV1().to(device)
    ck=torch.load(args.init,map_location='cpu',weights_only=False); model.load_state_dict(ck['model'])
    params=sum(p.numel() for p in model.parameters())
    print('device',device,'params',params,'train',len(train_ds),counts,'val',len(val_ds),val_counts,flush=True)
    ce=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); scaler=torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))
    start_ep=0; best_f1=-1; log=[]
    if args.resume:
        r=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(r['model']); opt.load_state_dict(r['optimizer']); start_ep=r.get('epoch',-1)+1; best_f1=r.get('best_f1',-1); log=r.get('log',[])
    for ep in range(start_ep,args.epochs):
        freeze = ep < args.freeze_backbone_epochs
        for name,p in model.named_parameters():
            p.requires_grad = True
            if freeze and not name.startswith('fc.'):
                p.requires_grad=False
        # refresh optimizer when freeze state changes
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr*(0.5 if freeze else 1.0),weight_decay=1e-4)
        model.train(); t0=time.time(); losses=[]
        for bi,(x,y) in enumerate(train_loader):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                logits=model(x); loss=ce(logits,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.item()))
            if (bi+1)%100==0: print(f'ep {ep} step {bi+1}/{len(train_loader)} loss {np.mean(losses[-100:]):.4f}',flush=True)
        val=evaluate(model,val_loader,device)
        f1=val['th0.5']['f1']; rec={'epoch':ep,'freeze_backbone':freeze,'train_loss':float(np.mean(losses)),'val':val,'time_sec':time.time()-t0}
        log.append(rec); print(json.dumps(rec),flush=True)
        save={'model':model.state_dict(),'optimizer':opt.state_dict(),'args':vars(args),'epoch':ep,'best_f1':best_f1,'log':log,'params':params}
        torch.save(save,outdir/'last.pt'); torch.save(save,outdir/f'epoch_{ep:03d}.pt')
        if f1>best_f1:
            best_f1=f1; save['best_f1']=best_f1; torch.save(save,outdir/'best.pt')
        (outdir/'log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    model.eval(); dummy=torch.zeros(1,3,args.img_size,args.img_size,device=device)
    torch.onnx.export(model,dummy,str(outdir/'vww_finetuned.onnx'),input_names=['images'],output_names=['logits'],opset_version=12)
    print('saved',outdir,flush=True)
if __name__=='__main__': main()
