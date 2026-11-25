"""Single held-out evaluation of the validation-selected pure 640 baseline."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
from pycocotools.coco import COCO
from run import STAGE, REF, DATA_YAML, reference, proof, sha, dump, now

def fixed_operating_point(evaluator):
    i=int(np.argmin(np.abs(evaluator.params.iouThrs-.5)))
    totals={int(c):[0,0,0] for c in evaluator.params.catIds}
    for item in evaluator.evalImgs:
        if item is None or list(item["aRng"])!=list(evaluator.params.areaRng[0]):continue
        keep=(np.asarray(item["dtScores"])>=.25) & ~item["dtIgnore"][i].astype(bool)
        match=item["dtMatches"][i]>0
        v=totals[int(item["category_id"])]
        v[0]+=int((keep & match).sum())
        v[1]+=int((keep & ~match).sum())
        v[2]+=int((~item["gtIgnore"].astype(bool)).sum())
    def metrics(tp,fp,gt):
        assert 0<=tp<=gt
        return dict(TP=tp,FP=fp,FN=gt-tp,GT=gt,P=tp/(tp+fp) if tp+fp else None,R=tp/gt if gt else None)
    cls={evaluator.cocoGt.cats[k]["name"]:metrics(*v) for k,v in totals.items()}
    return dict(score=.25,iou=.5,maxDets=100,micro=metrics(*map(sum,zip(*totals.values()))),classes=cls)

def main():
    folder=proof("C0")
    lock=json.loads((STAGE/"evaluation_plan.json").read_text())
    assert lock["test_checkpoint"]=="C0 Val-selected actual BEST only"
    assert lock["fixed_operating_point"]=={"score":.25,"iou":.5,"coco_maxDets":100}
    ck=STAGE/"runs/C0/weights/best.pt"
    selected=json.loads((folder/"evaluation/summary.json").read_text())["best"]
    decision=json.loads((STAGE/"decision.json").read_text())
    assert selected["sha256"]==decision["baseline_best_sha256"]==sha(ck)
    end=folder/"test"; result=end/"metrics.json"
    if result.exists():
        cached=json.loads(result.read_text()); assert cached["sha256"]==sha(ck)
        print(json.dumps(cached["main"])); return
    end.mkdir(exist_ok=True)
    dump(end/"selection_receipt.json",dict(time=now(),checkpoint=str(ck),sha256=sha(ck),
         selected_using="Val only",decision_sha256=sha(STAGE/"decision.json"),
         plan_sha256=sha(STAGE/"evaluation_plan.json"),validation_metrics=selected["main"]))
    dataset=reference.DATA
    gtpath=dataset/"annotations/instances_test.json"
    gt=COCO(str(gtpath))
    ids={Path(im["file_name"]).name:k for k,im in gt.imgs.items()}
    categories={v["name"]:k for k,v in gt.cats.items()}
    assert len(ids)==len(gt.imgs) and set(categories)==set(reference.NAMES)
    paths=sorted(dataset/"images/test"/name for name in ids)
    assert all(p.is_file() for p in paths) and len(paths)>0
    for p in paths:
        label=dataset/"labels/test"/(p.stem+".txt")
        expected=sum(not a.get("iscrowd",0) and not a.get("ignore",0) for a in gt.imgToAnns[ids[p.name]])
        lines=label.read_text().splitlines() if label.is_file() else []
        assert sum(bool(line.strip()) for line in lines)==expected, f"COCO/YOLO label count mismatch in {p.name}"
    for split in ("train","val"):
        known={Path(x).name for x in (REF/"data"/(split+".txt")).read_text().splitlines()}
        assert known.isdisjoint(ids), f"Filename overlap with {split}"
    txt=end/"test.txt";txt.write_text("".join(str(p)+"\n" for p in paths))
    test_yaml=end/"test.yaml"
    cfg=yaml.safe_load(DATA_YAML.read_text())
    cfg["test"]=str(txt)
    test_yaml.write_text(yaml.safe_dump(cfg,sort_keys=False))
    validator=reference.TaskValidator(args=dict(task="detect",data=str(test_yaml),imgsz=640,
        batch=32,workers=4,device="0",conf=.001,iou=.7,max_det=300,rect=True,save_json=True,
        plots=False,verbose=False,half=False,split="test"),save_dir=end)
    native=validator(model=reference.load_model(ck))
    assert validator.seen==len(ids)
    preds=[dict(image_id=ids[Path(row["file_name"]).name],category_id=categories[reference.NAMES[int(row["category_id"])-1]],
                bbox=row["bbox"],score=row["score"]) for row in validator.jdict]
    dump(end/"predictions.coco.json",preds)
    def summarize(ev):
        return dict(AP50=reference.ap(ev,iou=.50),AP75=reference.ap(ev,iou=.75),
                    AP50_95=reference.ap(ev),AR100=reference.ar100(ev))
    pooled=reference.coco_evaluate(gt,preds,[categories[n] for n in reference.NAMES])
    fixed=fixed_operating_point(pooled)
    val_gt=COCO(str(dataset/"annotations/instances_val.json"))
    val_cats={v["name"]:k for k,v in val_gt.cats.items()}
    val_pred=json.loads((folder/"evaluation/best/predictions.coco.json").read_text())
    val_fixed=fixed_operating_point(reference.coco_evaluate(val_gt,val_pred,[val_cats[n] for n in reference.NAMES]))
    metrics=summarize(pooled)
    metrics.update({f"AP_{size}":reference.ap(pooled,size) for size in ["small","medium","large"]})
    metrics.update(P=float(validator.metrics.box.mp),Recall=float(validator.metrics.box.mr))
    support=Counter(a["category_id"] for a in gt.anns.values() if not a.get("iscrowd",0))
    classes=[]
    indices=list(validator.metrics.box.ap_class_index)
    for i,n in enumerate(reference.NAMES):
        row=dict(name=n,GT=support[categories[n]],**summarize(reference.coco_evaluate(gt,preds,[categories[n]])))
        if i in indices:
            j=indices.index(i);row.update(P=float(validator.metrics.box.p[j]),Recall=float(validator.metrics.box.r[j]))
        classes.append(row)
    r=dict(time=now(),split="Test",images=len(ids),GT=len(gt.anns),predictions=len(preds),sha256=sha(ck),
           annotation_sha256=sha(gtpath),list_sha256=sha(txt),validation_selected_epoch=json.loads(
           (STAGE/"runs/C0/checkpoint_epoch_map.json").read_text())["best"],main=metrics,
           classes=classes,native=native,fixed_working_point=fixed,validation_fixed_working_point=val_fixed,
           evaluator="FP32 O2M class-aware NMS 0.7, COCO maxDets100. Native P/R use the validator F1 operating point on each split; fixed P/R uses the identical predeclared score 0.25 and IoU 0.50.")
    dump(result,r);print(json.dumps(r["main"],indent=2))

if __name__=="__main__": main()
