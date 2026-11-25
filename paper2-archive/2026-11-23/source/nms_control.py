"""Fixed-weight Val-only NMS control, used only if selected by the C0 diagnostic."""
import json
from pathlib import Path
from pycocotools.coco import COCO
from run import STAGE, DATA_YAML, reference, proof, sha, dump, now

def main():
    choice=json.loads((STAGE/"decision.json").read_text())
    ck=STAGE/"runs/C0/weights/best.pt"
    assert choice["arm"]=="NMS075" and sha(ck)==choice["baseline_best_sha256"]
    folder=proof("NMS075")/"evaluation/best"; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"metrics.json").is_file():
        assert json.loads((folder/"metrics.json").read_text())["sha256"]==sha(ck)
        return
    validator=reference.TaskValidator(args=dict(task="detect",data=str(DATA_YAML),imgsz=640,batch=32,
        workers=4,device="0",conf=.001,iou=.75,max_det=300,rect=True,save_json=True,plots=False,
        verbose=False,half=False,split="val"),save_dir=folder)
    native=validator(model=reference.load_model(ck));assert validator.seen==1000
    gt=COCO(str(reference.DATA/"annotations/instances_val.json"))
    imgs={Path(v["file_name"]).name:k for k,v in gt.imgs.items()}
    cats={v["name"]:k for k,v in gt.cats.items()}
    preds=[dict(image_id=imgs[Path(r["file_name"]).name],category_id=cats[reference.NAMES[int(r["category_id"])-1]],
                bbox=r["bbox"],score=r["score"]) for r in validator.jdict]
    dump(folder/"predictions.coco.json",preds)
    def score(ev):
        return dict(AP50=reference.ap(ev,iou=.5),AP75=reference.ap(ev,iou=.75),AP50_95=reference.ap(ev),AR100=reference.ar100(ev))
    pooled=reference.coco_evaluate(gt,preds,[cats[n] for n in reference.NAMES])
    main_metrics=score(pooled)
    main_metrics.update({f"AP_{s}":reference.ap(pooled,s) for s in ("small","medium","large")})
    main_metrics.update(P=float(validator.metrics.box.mp),Recall=float(validator.metrics.box.mr))
    rows=[dict(name=n,**score(reference.coco_evaluate(gt,preds,[cats[n]]))) for n in reference.NAMES]
    dump(folder/"metrics.json",dict(time=now(),split="Val",sha256=sha(ck),nms_iou=.75,
         checkpoint=str(ck),main=main_metrics,classes=rows,native=native,images=validator.seen))
if __name__=="__main__":main()
