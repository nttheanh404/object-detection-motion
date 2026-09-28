#!/usr/bin/env python3
from pathlib import Path
import argparse, json, random, math, time, cv2, numpy as np


def read_rows(tsv):
    rows=[]
    for line in Path(tsv).read_text().splitlines():
        p=line.split('\t')
        if len(p)>=2: rows.append((p[0],p[1]))
    return rows

def read_boxes_yolo(label_path,W,H):
    boxes=[]
    p=Path(label_path)
    if not p.exists(): return boxes
    for line in p.read_text(errors='ignore').splitlines():
        ps=line.split()
        if len(ps)<5: continue
        try:
            cls=int(float(ps[0])); cx=float(ps[1]); cy=float(ps[2]); bw=float(ps[3]); bh=float(ps[4])
        except Exception:
            continue
        if cls!=0: continue
        x1=(cx-bw/2)*W; y1=(cy-bh/2)*H; x2=(cx+bw/2)*W; y2=(cy+bh/2)*H
        x1=max(0,min(W,x1)); x2=max(0,min(W,x2)); y1=max(0,min(H,y1)); y2=max(0,min(H,y2))
        if x2>x1 and y2>y1: boxes.append([x1,y1,x2,y2])
    return boxes

def area(b): return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def inter(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    return max(0,x2-x1)*max(0,y2-y1)
def max_cov_ratios(crop, boxes):
    if not boxes: return 0.0,0.0
    ca=area(crop); max_cov=0.0; max_cr=0.0
    for b in boxes:
        ia=inter(crop,b)
        if ia<=0: continue
        max_cov=max(max_cov,ia/max(1e-6,area(b)))
        max_cr=max(max_cr,ia/max(1e-6,ca))
    return max_cov,max_cr

def clip_box(cx,cy,w,h,W,H):
    w=min(w,W); h=min(h,H)
    x1=cx-w/2; y1=cy-h/2; x2=cx+w/2; y2=cy+h/2
    if x1<0: x2-=x1; x1=0
    if y1<0: y2-=y1; y1=0
    if x2>W: x1-=x2-W; x2=W
    if y2>H: y1-=y2-H; y2=H
    x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
    return [int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))]

def sample_positive_for_box(b,W,H,rng,max_per_box=5,attempts=50):
    bx1,by1,bx2,by2=b; bw=bx2-bx1; bh=by2-by1; bcx=(bx1+bx2)/2; bcy=(by1+by2)/2
    frac=(bw*bh)/max(1,W*H)
    min_w=10.0/72.0*W; min_h=20.0/44.0*H
    n=min(max_per_box, 5 if frac<0.01 else (4 if frac<0.05 else 3))
    crops=[]; seen=set()
    for _ in range(attempts):
        ch=max(min_h,bh*rng.uniform(1.35,3.4))
        aspect=rng.uniform(0.42,0.80)
        cw=max(min_w,ch*aspect,bw*rng.uniform(1.25,2.8))
        if rng.random()<0.18:
            ch*=rng.uniform(1.1,1.35); cw*=rng.uniform(1.1,1.35)
        c=clip_box(bcx+rng.uniform(-0.30,0.30)*bw,bcy+rng.uniform(-0.40,0.20)*bh,cw,ch,W,H)
        if c[2]-c[0]<16 or c[3]-c[1]<16: continue
        cov,cr=max_cov_ratios(c,[b])
        if cov>=0.60 and cr>=0.025:
            key=tuple(v//4 for v in c)
            if key not in seen:
                seen.add(key); crops.append(c)
                if len(crops)>=n: break
    return crops

def sample_negative_clean(W,H,boxes,rng,target_n,attempts=60):
    crops=[]; seen=set()
    # Fast path for no-person images.
    min_w=8.0/72.0*W; max_w=18.0/72.0*W
    min_h=16.0/44.0*H; max_h=32.0/44.0*H
    for _ in range(attempts):
        ch=rng.uniform(min_h,max_h); aspect=rng.uniform(0.42,0.85); cw=float(np.clip(ch*aspect,min_w,max_w))
        if cw>=W or ch>=H:
            scale=min((W-2)/cw,(H-2)/ch,1.0); cw*=scale; ch*=scale
        if W-cw <= 1 or H-ch <= 1: continue
        cx=rng.uniform(cw/2,W-cw/2); cy=rng.uniform(ch/2,H-ch/2)
        c=clip_box(cx,cy,cw,ch,W,H)
        key=tuple(v//4 for v in c)
        if key in seen: continue
        maxcov,cr=max_cov_ratios(c,boxes)
        if maxcov==0.0 and cr==0.0:
            seen.add(key); crops.append(c)
            if len(crops)>=target_n: break
    return crops

def write_crop(img,crop,out_path,img_size=96,augment=False,rng=None):
    x1,y1,x2,y2=crop; c=img[y1:y2,x1:x2]
    if c.size==0: return False
    if augment and rng is not None:
        if rng.random()<0.5: c=cv2.flip(c,1)
        if rng.random()<0.7:
            c=np.clip(c.astype(np.float32)*rng.uniform(0.90,1.10)+rng.uniform(-8,8),0,255).astype(np.uint8)
        if rng.random()<0.20:
            c=np.clip(c.astype(np.float32)+np.random.normal(0,3,c.shape).astype(np.float32),0,255).astype(np.uint8)
    c=cv2.resize(c,(img_size,img_size),interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    return bool(cv2.imwrite(str(out_path),c,[int(cv2.IMWRITE_JPEG_QUALITY),92]))

def process_split(name, rows, outdir, rng, args):
    split_dir=outdir/name; man_path=outdir/f'{name}_manifest.tsv'
    if man_path.exists(): man_path.unlink()
    stats={'images':0,'images_read':0,'images_with_boxes':0,'boxes':0,'positive':0,'negative':0,'ambiguous_skipped':0,'skipped_no_img':0,'start_time':time.time()}
    rows=rows[:args.max_train_images] if name=='train' and args.max_train_images else rows
    rows=rows[:args.max_val_images] if name=='val' and args.max_val_images else rows
    rng.shuffle(rows)
    with man_path.open('w') as mf:
        for idx,(imgp,labp) in enumerate(rows):
            stats['images']+=1
            img=cv2.imread(imgp)
            if img is None:
                stats['skipped_no_img']+=1; continue
            H,W=img.shape[:2]; stats['images_read']+=1
            boxes=read_boxes_yolo(labp,W,H); stats['boxes']+=len(boxes)
            if boxes: stats['images_with_boxes']+=1
            pos=[]
            for b in boxes:
                pos.extend(sample_positive_for_box(b,W,H,rng,args.max_pos_per_box))
                if len(pos)>=args.max_pos_per_image: break
            pos=pos[:args.max_pos_per_image]
            neg_target=max(2,int(round(len(pos)*args.neg_ratio))) if pos else args.neg_per_empty_image
            neg=sample_negative_clean(W,H,boxes,rng,neg_target,args.neg_attempts)
            stem=f'{idx:07d}'
            for j,c in enumerate(pos):
                out=split_dir/'person'/f'{stem}_p{j:02d}.jpg'
                if write_crop(img,c,out,args.img_size,augment=(name=='train'),rng=rng):
                    mf.write(f'{out}\t1\t{imgp}\t{labp}\t{c[0]},{c[1]},{c[2]},{c[3]}\n'); stats['positive']+=1
            for j,c in enumerate(neg):
                out=split_dir/'non_person'/f'{stem}_n{j:02d}.jpg'
                if write_crop(img,c,out,args.img_size,augment=(name=='train'),rng=rng):
                    mf.write(f'{out}\t0\t{imgp}\t{labp}\t{c[0]},{c[1]},{c[2]},{c[3]}\n'); stats['negative']+=1
            if idx and idx%1000==0:
                mf.flush()
                print(f'{name} {idx}/{len(rows)} pos={stats["positive"]} neg={stats["negative"]}', flush=True)
    stats['manifest']=str(man_path); stats['elapsed_sec']=time.time()-stats['start_time']
    return stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/silu10_grid_train/train.tsv')
    ap.add_argument('--val',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/silu10_grid_train/val.tsv')
    ap.add_argument('--outdir',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/person_crop_classifier_dataset_v1')
    ap.add_argument('--seed',type=int,default=123); ap.add_argument('--img-size',type=int,default=96)
    ap.add_argument('--neg-ratio',type=float,default=1.5); ap.add_argument('--neg-attempts',type=int,default=60); ap.add_argument('--neg-per-empty-image',type=int,default=2)
    ap.add_argument('--max-pos-per-box',type=int,default=5); ap.add_argument('--max-pos-per-image',type=int,default=30)
    ap.add_argument('--max-train-images',type=int,default=16000); ap.add_argument('--max-val-images',type=int,default=3000)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    rng=random.Random(args.seed); np.random.seed(args.seed)
    tr=read_rows(args.train); va=read_rows(args.val)
    train_stats=process_split('train',tr,outdir,rng,args)
    val_stats=process_split('val',va,outdir,rng,args)
    summary={'args':vars(args),'train':train_stats,'val':val_stats,'rules':{'positive':'coverage with one person bbox >=60% and person crop ratio >=2.5%','negative':'zero intersection with every person bbox','ambiguous':'not sampled','crop':'random around 12x24 grid-window geometry then resized to 96x96'}}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    md=['# Person crop classifier dataset v1','',f'Output: `{outdir}`','', '| split | images read | images with boxes | boxes | positive | negative | pos:neg | elapsed |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split,st in [('train',train_stats),('val',val_stats)]:
        md.append(f"| {split} | {st['images_read']} | {st['images_with_boxes']} | {st['boxes']} | {st['positive']} | {st['negative']} | 1:{st['negative']/max(1,st['positive']):.2f} | {st['elapsed_sec']:.1f}s |")
    md += ['', 'Manifests:', '', f"- train: `{train_stats['manifest']}`", f"- val: `{val_stats['manifest']}`"]
    (outdir/'summary.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
