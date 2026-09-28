#!/usr/bin/env python3
from pathlib import Path
import argparse, json, time, random
import cv2, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models

class CropDataset(Dataset):
    def __init__(self, manifest, img_size=96, train=False):
        self.rows=[]; self.img_size=img_size; self.train=train
        for line in Path(manifest).read_text().splitlines():
            p=line.split('\t')
            if len(p)>=2:
                self.rows.append((p[0],int(p[1])))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        path,label=self.rows[idx]
        img=cv2.imread(path)
        if img is None:
            img=np.zeros((self.img_size,self.img_size,3),np.uint8)
        else:
            img=cv2.resize(img,(self.img_size,self.img_size),interpolation=cv2.INTER_AREA)
        if self.train:
            if random.random()<0.5: img=cv2.flip(img,1)
            if random.random()<0.25:
                img=np.clip(img.astype(np.float32)*random.uniform(0.95,1.05)+random.uniform(-5,5),0,255).astype(np.uint8)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        mean=np.array([0.485,0.456,0.406],np.float32); std=np.array([0.229,0.224,0.225],np.float32)
        img=(img-mean)/std
        return torch.from_numpy(img.transpose(2,0,1)).float(), torch.tensor(label,dtype=torch.long)

def make_model(arch, pretrained=True, num_classes=2):
    if arch=='shufflenet_v2_x0_5':
        weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        m=models.shufflenet_v2_x0_5(weights=weights)
        m.fc=nn.Linear(m.fc.in_features,num_classes)
        return m
    if arch=='mobilenet_v3_small':
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m=models.mobilenet_v3_small(weights=weights)
        m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,num_classes)
        return m
    raise ValueError(arch)

def evaluate(model, loader, device, thresholds=(0.3,0.5,0.7)):
    model.eval(); losses=[]; ce=nn.CrossEntropyLoss(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device)
            logits=model(x); loss=ce(logits,y); losses.append(float(loss.item()))
            prob=torch.softmax(logits,dim=1)[:,1]
            ys.append(y.cpu().numpy()); ps.append(prob.cpu().numpy())
    y=np.concatenate(ys) if ys else np.zeros((0,),np.int64); p=np.concatenate(ps) if ps else np.zeros((0,),np.float32)
    out={'loss':float(np.mean(losses)) if losses else None,'n':int(len(y)),'pos_frac':float(y.mean()) if len(y) else None}
    for th in thresholds:
        pred=p>=th
        tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
        out[f'th{th}']={'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),'fp':fp,'fn':fn,'tp':tp,'tn':tn,'pred_pos_frac':float(pred.mean()) if len(pred) else 0.0}
    return out

def class_counts(ds):
    c=[0,0]
    for _,y in ds.rows: c[y]+=1
    return c

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True); ap.add_argument('--val',required=True); ap.add_argument('--outdir',required=True)
    ap.add_argument('--arch',default='shufflenet_v2_x0_5'); ap.add_argument('--pretrained',action='store_true')
    ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch',type=int,default=128); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--img-size',type=int,default=96)
    ap.add_argument('--resume',default=''); ap.add_argument('--no-balanced-sampler',action='store_true')
    args=ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    train_ds=CropDataset(args.train,args.img_size,train=True); val_ds=CropDataset(args.val,args.img_size,train=False)
    counts=class_counts(train_ds); val_counts=class_counts(val_ds)
    print('device',device,'train',len(train_ds),counts,'val',len(val_ds),val_counts,flush=True)
    if args.no_balanced_sampler:
        train_loader=DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=args.workers,pin_memory=device.startswith('cuda'))
    else:
        weights=[1.0/max(1,counts[y]) for _,y in train_ds.rows]
        sampler=WeightedRandomSampler(weights,num_samples=len(weights),replacement=True)
        train_loader=DataLoader(train_ds,batch_size=args.batch,sampler=sampler,num_workers=args.workers,pin_memory=device.startswith('cuda'))
    val_loader=DataLoader(val_ds,batch_size=args.batch*2,shuffle=False,num_workers=args.workers,pin_memory=device.startswith('cuda'))
    model=make_model(args.arch,args.pretrained).to(device)
    total=sum(p.numel() for p in model.parameters())
    print('params',total,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,args.epochs))
    scaler=torch.cuda.amp.GradScaler(enabled=device.startswith('cuda'))
    ce=nn.CrossEntropyLoss()
    start_ep=0; best_f1=-1; log=[]
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); start_ep=ck.get('epoch',-1)+1; best_f1=ck.get('best_f1',-1); log=ck.get('log',[])
    for ep in range(start_ep,args.epochs):
        model.train(); t0=time.time(); losses=[]
        for bi,(x,y) in enumerate(train_loader):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
                logits=model(x); loss=ce(logits,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.item()))
            if (bi+1)%100==0: print(f'ep {ep} step {bi+1}/{len(train_loader)} loss {np.mean(losses[-100:]):.4f}',flush=True)
        sched.step(); val=evaluate(model,val_loader,device)
        f1=val['th0.5']['f1']; rec={'epoch':ep,'train_loss':float(np.mean(losses)),'val':val,'time_sec':time.time()-t0,'lr':sched.get_last_lr()[0]}
        log.append(rec); print(json.dumps(rec),flush=True)
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'args':vars(args),'epoch':ep,'best_f1':best_f1,'log':log,'params':total}
        torch.save(ck,outdir/'last.pt')
        torch.save(ck,outdir/f'epoch_{ep:03d}.pt')
        if f1>best_f1:
            best_f1=f1; ck['best_f1']=best_f1; torch.save(ck,outdir/'best.pt')
        (outdir/'log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    model.eval(); dummy=torch.zeros(1,3,args.img_size,args.img_size,device=device)
    torch.onnx.export(model,dummy,str(outdir/'person_crop_classifier.onnx'),input_names=['images'],output_names=['logits'],opset_version=12)
    print('saved',outdir,flush=True)
if __name__=='__main__': main()
