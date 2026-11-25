"""Val-only audit of raw O2M anchors, scores, native class-aware NMS, caps and input shifts."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import yaml
from pycocotools.coco import COCO

def iou(boxes, target):
    lo=np.maximum(boxes[:,:2],target[:2]); hi=np.minimum(boxes[:,2:],target[2:])
    xy=np.maximum(hi-lo,0); intersection=xy[:,0]*xy[:,1]
    area=np.maximum(boxes[:,2]-boxes[:,0],0)*np.maximum(boxes[:,3]-boxes[:,1],0)
    truth=max(target[2]-target[0],0)*max(target[3]-target[1],0)
    return intersection/np.maximum(area+truth-intersection,1e-12)

# run sets the import paths for the frozen remote source and its pickle classes.
from run import STAGE,REF,DATA_YAML,BASE_INIT,reference,proof,sha,dump,now
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

def metrics_for_gt(target,co,boxes,scores,levels,kept,labels,pre_count,nms_boxes):
    q=iou(boxes,target); best=int(q.argmax()); sc=scores[:,co]
    r=dict(raw_max_iou=float(q[best]),raw_best_level=int(levels[best]))
    for level in (2,3,4,5): r[f"p{level}_max_iou"]=float(q[levels==level].max())
    for cut,suffix in ((.50,"50"),(.75,"75")):
        good=q>=cut
        hits=np.flatnonzero(good)
        correct=(labels==co)
        valid=correct & good[kept]
        in_top300=np.arange(len(kept))<300
        in_top100=np.cumsum(correct)<=100
        reason="retained"
        if not len(hits): reason="raw_geometry"
        else:
            b=int(hits[np.argmax(sc[hits])]); score=float(sc[b])
            r["score"+suffix]=score
            r["score_rank_image_class"+suffix]=int(np.count_nonzero(sc>score)+1)
            r["class_rank_at_anchor"+suffix]=int(np.count_nonzero(scores[b]>score)+1)
            r["score_best_level"+suffix]=int(levels[b])
            r["anchor"+suffix]=b
            r["score_best_iou"+suffix]=float(q[b])
            r["p2_supported"+suffix]=bool(np.any(good & (levels==2)))
            r["p3_supported"+suffix]=bool(np.any(good & (levels==3)))
            for scale in (2,3):
                eligible=hits[levels[hits]==scale]
                if len(eligible):
                    pos=int(eligible[np.argmax(sc[eligible])])
                    r[f"p{scale}_score"+suffix]=float(sc[pos])
                    r[f"p{scale}_class_rank"+suffix]=int(np.count_nonzero(scores[pos]>sc[pos])+1)
            if score<=.001: reason="correct_class_below_floor"
            elif not valid.any(): reason="precap_or_nms" if pre_count>30000 else "nms"
            elif not (valid & in_top300).any(): reason="image_cap300"
            elif not (valid & in_top300 & in_top100).any(): reason="class_cap100"
            if valid.any():
                ix=int(np.flatnonzero(valid)[0])
                r["retained_score"+suffix]=float(sc[kept[ix]])
                r["retained_level"+suffix]=int(levels[kept[ix]])
                r["retained_iou"+suffix]=float(q[kept[ix]])
                r["retained_rank_in_image_class"+suffix]=int(correct[:ix+1].sum())
            if reason=="nms":
                suppressors=np.flatnonzero(correct & (sc[kept]>=score))
                if len(suppressors):
                    overlap=iou(nms_boxes[kept[suppressors]],nms_boxes[b])
                    if overlap.max()>.7:
                        j=suppressors[np.argmax(overlap)]
                        r["overlap_witness_pair_iou"+suffix]=float(overlap.max())
                        r["overlap_witness_iou_gt"+suffix]=float(q[kept[j]])
                        r["overlap_witness_score"+suffix]=float(sc[kept[j]])
        r["stage"+suffix]=reason
    return r

class Audit(reference.TaskValidator):
    def preprocess(self,batch):
        batch=super().preprocess(batch)
        dx,dy=self.shift
        if dx or dy:
            x=batch["img"]; h,w=x.shape[-2:]
            shifted=torch.full_like(x,114/255)
            shifted[...,dy:,dx:]=x[...,:h-dy,:w-dx]
            batch["img"]=shifted
        return batch

    def postprocess(self,pred):
        decoded,meta=pred
        assert not self.end2end and decoded.shape[1]==11 and len(meta["feats"])==4
        y=decoded.detach().clone()
        dx,dy=self.shift
        y[:,0]-=dx; y[:,1]-=dy
        level=np.concatenate([np.full(int(v.shape[-2]*v.shape[-1]),k,np.int8) for k,v in enumerate(meta["feats"],2)])
        assert len(level)==y.shape[2]
        # Same NMS as the official validator. Keep the tail temporarily to distinguish suppression from max_det.
        detections,indices=non_max_suppression(y.clone(),.001,.7,multi_label=True,agnostic=False,
                                              nc=7,max_det=30000,max_time_img=5.0,return_idxs=True)
        self.captured=(ops.xywh2xyxy(y[:,:4].transpose(1,2)),y[:,4:].transpose(1,2),level,detections,indices)
        return [dict(bboxes=d[:300,:4],conf=d[:300,4],cls=d[:300,5],extra=d[:300,6:]) for d in detections]

    def update_metrics(self,preds,batch):
        raw_boxes,raw_scores,level,dets,idxs=self.captured
        for si,name in enumerate(batch["im_file"]):
            pb=self._prepare_batch(si,batch)
            image_id=self.img_ids[Path(name).name]
            h,w=pb["ori_shape"]
            b=ops.scale_boxes(pb["imgsz"],raw_boxes[si].clone(),pb["ori_shape"],pb["ratio_pad"]).cpu().numpy()
            scores=raw_scores[si].cpu().numpy()
            nms_boxes=raw_boxes[si].detach().cpu().numpy()
            kin=idxs[si].detach().cpu().numpy().reshape(-1).astype(int)
            labels=dets[si][:,5].detach().cpu().numpy().astype(int)
            if len(kin):
                assert np.max(np.abs(scores[kin,labels]-dets[si][:,4].cpu().numpy()))<1e-6
            pre_count=int(np.count_nonzero(scores>.001))
            real=self.gt.imgToAnns[image_id]
            truths=[]
            for ann in real:
                if ann.get("iscrowd",0) or ann.get("ignore",0): continue
                cn=self.gt.cats[ann["category_id"]]["name"]
                co=reference.NAMES.index(cn)
                x,y,ww,hh=ann["bbox"]
                truth=np.array([x,y,x+ww,y+hh],dtype=np.float32)
                gain=pb["ratio_pad"][0][0]
                row=dict(image=Path(name).name,image_id=image_id,ann_id=ann["id"],cls=cn,
                         short640=float(min(ww,hh)*gain),area=ann.get("area",ww*hh),
                         inner=(min(x,y,w-x-ww,h-y-hh)*gain)>2.5)
                row.update(metrics_for_gt(truth,co,b,scores,level,kin,labels,pre_count,nms_boxes))
                self.rows.append(row);truths.append((co,truth))
            if self.shift==(0,0):
                for pos in range(min(300,len(kin))):
                    k=kin[pos];c=labels[pos];s=float(scores[k,c])
                    if s<.25: break
                    same=max((float(iou(b[k:k+1],t)[0]) for kcls,t in truths if kcls==c),default=0.)
                    other=max((float(iou(b[k:k+1],t)[0]) for kcls,t in truths if kcls!=c),default=0.)
                    self.detections.append(dict(image_id=image_id,image=Path(name).name,cls=reference.NAMES[c],
                        level=int(level[k]),score=s,iou_same=same,iou_other=other,box=b[k].tolist()))
        self.captured=None
        super().update_metrics(preds,batch)

def group(rows):
    if not rows: return {"n":0}
    stages={c:dict(Counter(r["stage"+c] for r in rows)) for c in ("50","75")}
    res=dict(n=len(rows),stage50=stages["50"],stage75=stages["75"])
    for key in ["raw_max_iou","p2_max_iou","p3_max_iou","score75","score_rank_image_class75",
                "class_rank_at_anchor75","retained_score75"]:
        vs=[r[key] for r in rows if key in r]
        res[key+"_median"]=float(np.median(vs)) if vs else None
    res["best_level75"]=dict(Counter(r.get("score_best_level75","none") for r in rows))
    res["p2_has_75"]=sum(r.get("p2_supported75",False) for r in rows)
    res["p3_has_75"]=sum(r.get("p3_supported75",False) for r in rows)
    res["low_025_qualified_75"]=sum(r.get("score75",1)<.25 for r in rows)
    res["qualified_75_class_not_top1"]=sum(r.get("class_rank_at_anchor75",1)>1 for r in rows)
    return res

def groups(rows):
    return {"all":rows,**{c:[r for r in rows if r["cls"]==c] for c in reference.NAMES},
         "narrow_lt8":[r for r in rows if r["short640"]<8],
         "narrow_lt16":[r for r in rows if r["short640"]<16],
         "LC_TC_lt16":[r for r in rows if r["cls"] in ("LC","TC") and r["short640"]<16],
         "short_ge32":[r for r in rows if r["short640"]>=32],
         "coco_large":[r for r in rows if r["area"]>=96**2]}

def compare(base,shifted):
    byid={r["ann_id"]:r for r in shifted}
    assert {r["ann_id"] for r in base}==set(byid)
    ans={}
    for k,gg in groups(base).items():
        rs=[r for r in gg if r["inner"]]
        if not rs: continue
        d=np.array([byid[r["ann_id"]]["raw_max_iou"]-r["raw_max_iou"] for r in rs])
        before=np.array([r["raw_max_iou"]>=.75 for r in rs])
        after=np.array([byid[r["ann_id"]]["raw_max_iou"]>=.75 for r in rs])
        ans[k]=dict(n=len(rs),base75=int(before.sum()),shifted75=int(after.sum()),lost=int((before & ~after).sum()),
                    gained=int((~before & after).sum()),mean_iou_delta=float(d.mean()),mean_abs_iou_delta=float(np.abs(d).mean()))
    return ans

def write_jsonl(path,rows):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    tmp.replace(path)

def attach_coco_matches(ev,rows):
    by_id={r["ann_id"]:r for r in rows}
    seen=set()
    for item in ev.evalImgs:
        if item is None or list(item["aRng"])!=list(ev.params.areaRng[0]): continue
        for j,gid in enumerate(item["gtIds"]):
            if item["gtIgnore"][j]: continue
            r=by_id[int(gid)]
            assert gid not in seen
            seen.add(gid)
            for cut,suf in ((.5,"50"),(.75,"75")):
                k=int(np.abs(ev.params.iouThrs-cut).argmin())
                match=int(item["gtMatches"][k,j])
                r["coco_match_score"+suf]=float(ev.cocoDt.anns[match]["score"]) if match else None
    assert seen==set(by_id)
    answer={}
    for key,group_rows in groups(rows).items():
        missed=[r for r in group_rows if (r["coco_match_score75"] or 0.)<.25]
        credible=[r for r in missed if r.get("p2_score75",0.)>.001]
        answer[key]=dict(n=len(group_rows),matched75_official=sum(r["coco_match_score75"] is not None for r in group_rows),
              matched75_at025=len(group_rows)-len(missed),miss_at025=len(missed),
              miss_at025_by_stage75=dict(Counter(r["stage75"] for r in missed)),
              miss_at025_with_p2_box75=sum(r.get("p2_supported75",False) for r in missed),
              miss_at025_with_p2_score_gt_floor=len(credible),
              miss_at025_with_p2_score_floor_to025=sum(r["p2_score75"]<.25 for r in credible),
              miss_at025_with_p2_score_gt_floor_wrong_top1=sum(r["p2_class_rank75"]>1 for r in credible))
    return answer

def main():
    p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");p.add_argument("--arm",default="C0")
    p.add_argument("--phase",choices=["all","base"],default="all");opts=p.parse_args()
    if opts.arm!="C0":import candidates
    if opts.smoke:
        ck=BASE_INIT;dest=STAGE/"diagnostic_smoke"; path=dest/"val.yaml";dest.mkdir(exist_ok=True)
        subset=dest/"subset.txt";subset.write_text("\n".join((REF/"data/val.txt").read_text().splitlines()[:2])+"\n")
        cfg=yaml.safe_load(DATA_YAML.read_text());cfg["val"]=str(subset);path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    else:
        ck=STAGE/"runs"/opts.arm/"weights/best.pt"
        selected=json.loads((proof(opts.arm)/"evaluation/summary.json").read_text())["best"]
        assert sha(ck)==selected["sha256"]
        dest=STAGE/"diagnostics"/opts.arm;path=DATA_YAML;dest.mkdir(parents=True,exist_ok=True)
    gt=COCO(str(reference.DATA/"annotations/instances_val.json"))
    ids={Path(v["file_name"]).name:k for k,v in gt.imgs.items()}
    cats={v["name"]:k for k,v in gt.cats.items()}
    shifts=[(0,0)] if opts.phase=="base" else [(0,0),(1,0),(0,1),(2,0),(0,2)]
    result=dict(time=now(),checkpoint_sha=sha(ck),split="Val",smoke=opts.smoke,
                interpretation="GT candidate existence allows reuse; score ranks descriptive; .25 is a diagnostic working point and not the COCO floor.",
                shifts={})
    for dx,dy in shifts:
        key=f"x{dx}_y{dy}";folder=dest/key;folder.mkdir(exist_ok=True)
        v=Audit(args=dict(task="detect",data=str(path),imgsz=640,batch=2 if opts.smoke else 32,
                     workers=0 if opts.smoke else 4,device="cpu" if opts.smoke else "0",
                     conf=.001,iou=.7,max_det=300,rect=True,save_json=True,plots=False,verbose=False,half=False,
                     split="val"),save_dir=folder)
        v.shift=(dx,dy);v.gt=gt;v.img_ids=ids;v.rows=[];v.detections=[]
        v(model=reference.load_model(ck))
        assert len({r["ann_id"] for r in v.rows})==len(v.rows)
        if not opts.smoke:
            assert v.seen==1000 and len(v.rows)==2612
        write_jsonl(folder/"gt.jsonl",v.rows)
        if key=="x0_y0":
            base=v.rows
            result["groups"]={k:group(g) for k,g in groups(base).items()}
            write_jsonl(folder/"detections_at_025.jsonl",v.detections)
            result["detections_at_025"]={}
            for n in ["all",*reference.NAMES]:
                d=[r for r in v.detections if n=="all" or r["cls"]==n]
                result["detections_at_025"][n]=dict(n=len(d),background_like=sum(max(a["iou_same"],a["iou_other"])<.10 for a in d),
                    wrong_class_overlap50=sum(a["iou_same"]<.5 and a["iou_other"]>=.5 for a in d))
            if not opts.smoke:
                rows=[dict(image_id=ids[r["file_name"]],category_id=cats[reference.NAMES[int(r["category_id"])-1]],
                     bbox=r["bbox"],score=r["score"]) for r in v.jdict]
                canonical=proof(opts.arm)/"evaluation/best/predictions.coco.json"
                official=json.loads(canonical.read_text())
                order=lambda r:(r["image_id"],r["category_id"],r["score"],*r["bbox"])
                same=sorted(rows,key=order)==sorted(official,key=order)
                ev=reference.coco_evaluate(gt,rows,[cats[n] for n in reference.NAMES])
                result["one_to_one"]=attach_coco_matches(ev,base)
                write_jsonl(folder/"gt.jsonl",base)
                replay=dict(AP50=reference.ap(ev,iou=.5),AP75=reference.ap(ev,iou=.75),AP50_95=reference.ap(ev))
                delta=max(abs(val-selected["main"][k]) for k,val in replay.items())
                result["official_replay"]=dict(exact_predictions=same,predictions=len(rows),official_count=len(official),
                         metrics=replay,max_abs_metric_delta=delta)
                assert delta<1e-4, result["official_replay"]
        else: result["shifts"][key]=compare(base,v.rows)
        dump(dest/"summary.json",result)
    result["complete"]=True;result["completed_at"]=now();dump(dest/"summary.json",result)

if __name__=="__main__": main()
