#!/usr/bin/env python3
from pathlib import Path
import json, cv2, numpy as np, argparse, importlib.util

sweep_path=Path('/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/sweep_motion_window_grid_demo.py')
spec=importlib.util.spec_from_file_location('sw', sweep_path)
sw=importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)

def containment(person,crop):
    px1,py1,px2,py2=person; cx1,cy1,cx2,cy2=crop
    ix1=max(px1,cx1); iy1=max(py1,cy1); ix2=min(px2,cx2); iy2=min(py2,cy2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    pa=max(1e-6,(px2-px1)*(py2-py1))
    return inter/pa

def bbox_motion_stats(person, active, W, H, grid_w, grid_h):
    x1,y1,x2,y2=person
    gx1=max(0,int(np.floor(x1*grid_w/W))); gy1=max(0,int(np.floor(y1*grid_h/H)))
    gx2=min(grid_w,int(np.ceil(x2*grid_w/W))); gy2=min(grid_h,int(np.ceil(y2*grid_h/H)))
    if gx2<=gx1 or gy2<=gy1: return 0,0,0.0
    roi=active[gy1:gy2,gx1:gx2]
    ac=int(roi.sum()); total=int(roi.size)
    return ac,total,ac/max(1,total)

def silu_native_boxes(summary_json):
    d=json.loads(Path(summary_json).read_text())
    out=[]
    for fr in d['frames']:
        boxes=[]
        for b in fr.get('boxes',[]):
            boxes.append([int(b['x1']),int(b['y1']),int(b['x2']),int(b['y2'])])
        out.append(boxes)
    return out

def expand_to_window(box,W,H,grid_w=72,grid_h=44,win_w=12,win_h=24):
    x1,y1,x2,y2=box
    cx=(x1+x2)/2; cy=(y1+y2)/2
    gx=cx*grid_w/W; gy=cy*grid_h/H
    # align to same 12x24 grid window, centered around heat component center
    gx1=int(round(gx-win_w/2)); gy1=int(round(gy-win_h/2))
    gx1=max(0,min(grid_w-win_w,gx1)); gy1=max(0,min(grid_h-win_h,gy1))
    return [int(gx1*W/grid_w), int(gy1*H/grid_h), int((gx1+win_w)*W/grid_w), int((gy1+win_h)*H/grid_h)]

def eval_proposals(name, proposals_by_frame, yolo_frames, eligible_by_frame, contain_thr=0.5):
    rows=[]
    for idx,props in enumerate(proposals_by_frame):
        persons=[b['xyxy'] for b in yolo_frames.get(idx,[])]
        elig=eligible_by_frame.get(idx,[False]*len(persons))
        all_hits=0; elig_hits=0
        for j,p in enumerate(persons):
            hit=max([containment(p,c) for c in props], default=0.0) >= contain_thr
            all_hits += int(hit)
            elig_hits += int(hit and (j < len(elig) and elig[j]))
        useful=sum(1 for c in props if any((j < len(elig) and elig[j]) and containment(p,c)>=contain_thr for j,p in enumerate(persons)))
        rows.append({'frame':idx,'persons':len(persons),'eligible':sum(elig),'props':len(props),'all_hits':all_hits,'eligible_hits':elig_hits,'useful':useful,'extra':len(props)-useful})
    total_y=sum(r['persons'] for r in rows); total_e=sum(r['eligible'] for r in rows)
    return {
        'name':name,
        'frames':len(rows),
        'total_yolo_persons':total_y,
        'total_motion_eligible_persons':total_e,
        'all_recall':sum(r['all_hits'] for r in rows)/max(1,total_y),
        'motion_eligible_recall':sum(r['eligible_hits'] for r in rows)/max(1,total_e),
        'avg_proposals':float(np.mean([r['props'] for r in rows])) if rows else 0,
        'avg_useful':float(np.mean([r['useful'] for r in rows])) if rows else 0,
        'avg_extra':float(np.mean([r['extra'] for r in rows])) if rows else 0,
        'max_proposals':int(max([r['props'] for r in rows], default=0)),
        'rows':rows,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/mot17_small/MOT17-02-FRCNN-raw.mp4')
    ap.add_argument('--yolo-json',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/person_classifier_probe/yolo_person_mot17_300.json')
    ap.add_argument('--vww-summary',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/person_classifier_probe/vww_motion_windows_vs_yolo_mot17/summary.json')
    ap.add_argument('--silu-strict',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/mot17_small/MOT17-02_student_silu26_grid_motion_demo_summary.json')
    ap.add_argument('--silu-loose',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/mot17_small/MOT17-02_student_silu26_grid_motion_demo_loose_summary.json')
    ap.add_argument('--outdir',default='/home/ai5070/babyalpha/robot_ws/benchmark_results/motion_detection_datasets/person_classifier_probe/fair_compare_silu26_vww')
    ap.add_argument('--max-frames',type=int,default=300)
    ap.add_argument('--contain-thr',type=float,default=0.5)
    ap.add_argument('--eligible-motion-cells',type=int,default=2)
    ap.add_argument('--eligible-motion-ratio',type=float,default=0.02)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    ydata=json.loads(Path(args.yolo_json).read_text())
    yolo_frames={r['frame']:r['boxes'] for r in ydata['frames']}
    cap=cv2.VideoCapture(args.video); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st=sw.make_motion_state(); eligible_by_frame={}
    for idx in range(args.max_frames):
        ok,fr=cap.read()
        if not ok: break
        score=sw.motion_score(fr,st)
        gs=cv2.resize(score,(72,44),interpolation=cv2.INTER_AREA)
        active=(gs>=0.30).astype(np.uint8)
        elig=[]
        for b in yolo_frames.get(idx,[]):
            mc,tc,mr=bbox_motion_stats(b['xyxy'],active,W,H,72,44)
            elig.append(mc>=args.eligible_motion_cells and mr>=args.eligible_motion_ratio)
        eligible_by_frame[idx]=elig
    cap.release()

    results=[]
    # VWW summary already evaluated with same yolo json/eligible definition; include compact forms.
    vww=json.loads(Path(args.vww_summary).read_text())['summary']
    for thr,d in vww.items():
        results.append({
            'name':f'VWW motion-window thr {thr}', 'params':'221k', 'method':'window 12x24 stride4 + VWW classifier',
            'all_recall':d['all_recall'], 'motion_eligible_recall':d['eligible_recall'],
            'avg_proposals':d['avg_passed_crops'], 'avg_useful':d['avg_useful_passed'], 'avg_extra':d['avg_extra_passed'], 'max_proposals':d['max_passed'],
            'total_yolo_persons':d['all_yolo_persons'], 'total_motion_eligible_persons':d['eligible_yolo_persons']
        })

    for label,path in [('Silu26 strict native',args.silu_strict),('Silu26 loose native',args.silu_loose)]:
        boxes=silu_native_boxes(path)[:args.max_frames]
        e=eval_proposals(label,boxes,yolo_frames,eligible_by_frame,args.contain_thr)
        results.append({k:v for k,v in e.items() if k!='rows'} | {'params':'1.65M','method':'native connected components on Silu26 heat×motion'})
        win_boxes=[[expand_to_window(b,W,H) for b in fr] for fr in boxes]
        e2=eval_proposals(label.replace('native','window12x24'),win_boxes,yolo_frames,eligible_by_frame,args.contain_thr)
        results.append({k:v for k,v in e2.items() if k!='rows'} | {'params':'1.65M','method':'same 12x24 crop window around Silu26 component center'})

    # sort manually: Silu then VWW? keep generated order but write table
    payload={'video':args.video,'yolo_json':args.yolo_json,'metric':{'containment_threshold':args.contain_thr,'motion_eligible':'YOLO person bbox is eligible if its own bbox contains >=2 active motion cells and active-cell ratio >=0.02 on common 72x44 motion grid','proposal_hit':'proposal/crop is a hit if it covers >=50% of YOLO bbox area'},'results':results}
    (outdir/'fair_compare.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    md=['# Fair compare: Silu26 heat-grid vs VWW motion-window classifier','',
        'Cùng video MOT17 300 frame, cùng YOLO person pseudo-label, cùng common motion grid `72×44`.','',
        'Metric công bằng ở đây: mọi thuật toán đều quy về **proposal/crop cuối cùng**. Một proposal đúng nếu nó phủ ít nhất `50%` diện tích bbox người YOLO. Người “có motion” là bbox YOLO có motion-cell bên trong theo cùng motion grid, không phụ thuộc Silu26 hay VWW.','',
        '| Method | Params | All YOLO recall | Moving-person recall | Avg proposals/frame | Avg useful | Avg extra/false | Max proposals |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        md.append(f"| {r['name']} | {r['params']} | {r['all_recall']*100:.1f}% | {r['motion_eligible_recall']*100:.1f}% | {r['avg_proposals']:.2f} | {r['avg_useful']:.2f} | {r['avg_extra']:.2f} | {r['max_proposals']} |")
    md += ['', '## Notes', '', '- Silu26 native giữ đúng output heatmap/component, nhưng component thường nhỏ nên bị thiệt nếu yêu cầu crop phủ phần lớn bbox người.', '- Silu26 window12x24 là bản chuẩn hóa sang crop giống hướng VWW. Cách này công bằng cho bài toán gửi crop/classify tiếp, nhưng nó làm mất một phần điểm mạnh localize gọn của heatmap.', '- VWW dùng motion window trước rồi classifier xác nhận person; output tự nhiên đã là crop nên metric proposal/crop hợp với bài toán cloud filter.']
    (outdir/'fair_compare.md').write_text('\n'.join(md),encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__': main()
