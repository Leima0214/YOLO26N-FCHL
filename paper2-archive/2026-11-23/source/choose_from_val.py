"""Choose one follow-up from completed, held-out validation diagnostics, before the C0 Test pass."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from run import STAGE, proof, sha, dump, now

def read_rows(folder):
    return [json.loads(s) for s in (folder/"gt.jsonl").read_text().splitlines()]

def narrow_geometry(base, folder):
    shifts = [{r["ann_id"]:r["raw_max_iou"] for r in read_rows(folder/s)}
              for s in ("x1_y0","x0_y1","x2_y0","x0_y2")]
    # Compare same-class narrow and wider GTs. Per-image sums keep co-occurring instances together.
    sums=defaultdict(lambda:np.zeros((2,2,3),dtype=float))
    for r in base:
        if r["cls"] not in ("LC","TC") or not r["inner"]: continue
        w=r["short640"]
        if 16<=w<32: continue
        group=sums[r["image"]][int(r["cls"]=="TC"),int(w>=32)]
        delta=np.mean([abs(s[r["ann_id"]]-r["raw_max_iou"]) for s in shifts])
        group += [1, r["raw_max_iou"]<.75, delta]
    if not sums: return {"sufficient":False}
    blocks=np.stack(list(sums.values()))
    counts=blocks.sum(0)[:,:,0]
    if not (counts>0).all():return {"sufficient":False,"counts_LC_TC__narrow_wide":counts.tolist()}
    weights=counts[:,0]/counts[:,0].sum()
    def differences(total):
        if np.any(total[:,:,0]==0):return None
        rates=total[:,:,1:]/total[:,:,:1]
        return weights@(rates[:,0]-rates[:,1])
    value=differences(blocks.sum(0))
    rng=np.random.default_rng(42)
    samples=[]
    for _ in range(1000):
        d=differences(blocks[rng.integers(len(blocks),size=len(blocks))].sum(0))
        if d is not None:samples.append(d)
    if not samples:return {"sufficient":False,"counts_LC_TC__narrow_wide":counts.tolist()}
    ci=np.percentile(samples,[2.5,97.5],axis=0).T
    return dict(sufficient=True,images=len(blocks),counts_LC_TC__narrow_wide=counts.tolist(),
                geometry_missing75_gap=float(value[0]),geometry_gap_interval=ci[0].tolist(),
                mean_abs_iou_shift_gap=float(value[1]),phase_gap_interval=ci[1].tolist(),
                strong_geometry_and_shift=bool((ci[:,0]>0).all()), valid_bootstrap_samples=len(samples),
                method="LC/TC within-class, weighted to narrow GT class mix, border excluded, narrow <16 vs wide >=32 input pixels, percentile bootstrap by Val image (1000 draws).")

def main():
    dest=STAGE/"decision.json"
    if dest.exists():
        cached=json.loads(dest.read_text())
        assert cached["baseline_best_sha256"]==sha(STAGE/"runs/C0/weights/best.pt")
        print(cached["arm"]);return
    folder=STAGE/"diagnostics/C0"
    summary=json.loads((folder/"summary.json").read_text())
    selected=json.loads((proof("C0")/"evaluation/summary.json").read_text())["best"]
    assert summary["complete"] and summary["split"]=="Val" and not summary["smoke"]
    assert summary["checkpoint_sha"]==selected["sha256"]==sha(STAGE/"runs/C0/weights/best.pt")
    assert summary["official_replay"]["max_abs_metric_delta"]<1e-4
    base=read_rows(folder/"x0_y0")
    assert len(base)==2612
    phase=narrow_geometry(base,folder)
    scoring=[r for r in base if r["stage75"]=="retained" and (r["coco_match_score75"] or 0.)<.25
             and r.get("score_best_level75")==2 and r.get("retained_level75")==2
             and .001<r.get("p2_score75",0.)<.25]
    nms=[r for r in base if r["stage75"]=="nms" and r["coco_match_score75"] is None
         and r.get("overlap_witness_pair_iou75",0.)>.7 and r.get("score75",0.)>=.25]
    signals=dict(p2_qualified_low_score_after_retention=len(scoring),
                 p2_images=len({r["image"] for r in scoring}),
                 nms_witness_ge025=len(nms),nms_images=len({r["image"] for r in nms}),
                 low_score_by_class={c:sum(r["cls"]==c for r in scoring) for c in ("LC","TC","AC","P","MC","LP","TP")})
    if phase.get("strong_geometry_and_shift"):
        arm="H1_L1"
        why="Within LC/TC, narrow boxes have both a larger raw IoU75 coverage deficit and greater shift sensitivity. Test one foldable linear L1 training residual; this alone does not identify irreversible information loss."
    elif signals["p2_images"]>1:
        arm="A"
        why="Across multiple Val images the strongest correct-class IoU75 candidate comes from P2, survives as a qualified candidate at the official floor, yet has a score below 0.25 and the GT is missed at that working point. Test a P2-only classification-logit residual with unchanged regression wiring."
    elif signals["nms_images"]>1:
        arm="NMS075"
        why="Multiple Val images have an IoU75-correct candidate scoring at least 0.25 removed by a retained overlapping box. Compare one fixed-weight NMS setting (0.75 vs 0.70) on Val."
    else:
        arm="H1_L1"
        why="P2 ranking and NMS lack replicated high-confidence qualifying examples under these definitions. Test one small, foldable L1 reparameterization as exploratory; the data do not establish a downsampling mechanism."
    dump(dest,dict(time=now(),arm=arm,reason=why,source="C0 Val only",baseline_best_sha256=selected["sha256"],
         diagnostic_sha256=sha(folder/"summary.json"),base_gt_sha256=sha(folder/"x0_y0/gt.jsonl"),
         decision_script_sha256=sha(Path(__file__)),baseline_validation=selected["main"],
         phase=phase,signals=signals))
    print(arm)

if __name__=="__main__": main()
