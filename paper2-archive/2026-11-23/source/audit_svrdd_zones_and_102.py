"""Zero-GPU audit of SVRDD Train/Val vertical zone distributions & 102 shared narrow failures."""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANN_DIR = Path("/SVRDD/SVRDD_YOLO_READY/annotations")
VAL_LIST = Path("/root/YOLO26N-FCHL/reports/paper2_cls_p2p3_snake_640_20261115_42264/data/val.txt")
STAGE = Path("/root/YOLO26N-FCHL/reports/paper2_cls_candidate_diagnosis_640_20261120_42106")
DIAG_DIR = STAGE / "diagnostics"


def classify_zone(y1: float, y2: float) -> str:
    if y2 <= 0.50:
        return "[0,50%]"
    if y1 >= 0.50 and y2 <= 0.75:
        return "[50,75%]"
    if y1 >= 0.75:
        return "[75,100%]"
    if y1 < 0.50 and y2 <= 0.75:
        return "cross_50%"
    if y1 >= 0.50 and y2 > 0.75:
        return "cross_75%"
    return "cross_50_75%"


def quantiles(vals):
    if not vals:
        return {}
    s = sorted(vals)
    n = len(s)
    qs = {}
    for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        idx = min(n - 1, int(round((p / 100.0) * (n - 1))))
        qs[f"p{p}"] = round(float(s[idx]), 4)
    return qs


def analyze_split(fname: str) -> dict:
    d = json.loads((ANN_DIR / fname).read_text())
    imgs = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}
    zones = Counter()
    narrow_zones = Counter()
    by_cls_zone = defaultdict(Counter)
    by_cls_total = Counter()
    by_cls_narrow = Counter()
    y1s, y2s = [], []
    ny1s, ny2s = [], []
    area_share = {"[0,50%]": 0.0, "[50,75%]": 0.0, "[75,100%]": 0.0}
    narrow_wh = Counter()

    for a in d["annotations"]:
        if a.get("iscrowd") or a.get("ignore"):
            continue
        im = imgs[a["image_id"]]
        w, h = im["width"], im["height"]
        _, y, bw, bh = a["bbox"]
        y1 = max(0.0, y / h)
        y2 = min(1.0, (y + bh) / h)
        y1s.append(y1)
        y2s.append(y2)
        cname = cats[a["category_id"]]
        short640 = min(bw, bh) * (640.0 / max(w, h))
        z = classify_zone(y1, y2)
        zones[z] += 1
        by_cls_zone[cname][z] += 1
        by_cls_total[cname] += 1

        # Vertical overlap fraction across the 3 regions
        total_h = max(1e-9, y2 - y1)
        area_share["[0,50%]"] += max(0.0, min(y2, 0.50) - max(y1, 0.0)) / total_h
        area_share["[50,75%]"] += max(0.0, min(y2, 0.75) - max(y1, 0.50)) / total_h
        area_share["[75,100%]"] += max(0.0, min(y2, 1.00) - max(y1, 0.75)) / total_h

        if short640 < 8.0:
            narrow_zones[z] += 1
            by_cls_narrow[cname] += 1
            ny1s.append(y1)
            ny2s.append(y2)
            if bh < bw:
                narrow_wh["h_lt_w"] += 1
            elif bw < bh:
                narrow_wh["w_lt_h"] += 1
            else:
                narrow_wh["eq"] += 1

    total = sum(zones.values())
    narrow_total = sum(narrow_zones.values())
    return dict(
        images=len(imgs),
        boxes=total,
        narrow_lt8_boxes=narrow_total,
        narrow_orientation=dict(narrow_wh),
        zones={k: dict(count=v, pct=round(100.0 * v / total, 2)) for k, v in zones.items()},
        narrow_zones={k: dict(count=v, pct=round(100.0 * v / narrow_total, 2)) for k, v in narrow_zones.items()},
        avg_vertical_overlap_pct={k: round(100.0 * v / total, 2) for k, v in area_share.items()},
        per_class={
            cn: dict(
                total=by_cls_total[cn],
                narrow_lt8=by_cls_narrow[cn],
                zones={
                    k: dict(count=by_cls_zone[cn][k], pct=round(100.0 * by_cls_zone[cn][k] / by_cls_total[cn], 2))
                    for k in ("[0,50%]", "cross_50%", "[50,75%]", "cross_75%", "[75,100%]", "cross_50_75%")
                    if by_cls_zone[cn][k]
                },
            )
            for cn in sorted(by_cls_total)
        },
        y1_quantiles=quantiles(y1s),
        y2_quantiles=quantiles(y2s),
        narrow_y1_quantiles=quantiles(ny1s),
        narrow_y2_quantiles=quantiles(ny2s),
    )


def load_gt_map(arm: str) -> dict:
    p = DIAG_DIR / arm / "x0_y0" / "gt.jsonl"
    out = {}
    with p.open() as f:
        for ln in f:
            r = json.loads(ln)
            out[int(r["ann_id"])] = r
    return out


def make_visual_collage(val_img_map, val_anns, val_imgs, cats, fail_cases, ctrl_cases, out_path: Path):
    font = ImageFont.load_default()
    cell_sz = 200
    hdr_h = 34
    gap = 6
    pair_w = cell_sz * 2 + gap
    pair_h = cell_sz + hdr_h
    cols = 3
    margin = 16
    section_h = 30
    fail_rows = (len(fail_cases) + cols - 1) // cols
    ctrl_rows = (len(ctrl_cases) + cols - 1) // cols
    canvas_w = margin * 2 + cols * pair_w + (cols - 1) * 14
    canvas_h = margin * 2 + section_h * 2 + (fail_rows + ctrl_rows) * (pair_h + 10) + 16
    canvas = Image.new("RGB", (canvas_w, canvas_h), (246, 248, 250))
    draw = ImageDraw.Draw(canvas)

    def render_pair(case, cx, cy, is_ctrl=False):
        ann = val_anns[case["ann_id"]]
        im_info = val_imgs[ann["image_id"]]
        fname = Path(im_info["file_name"]).name
        img_path = val_img_map[fname]
        bgr = cv2.imread(str(img_path))
        rgb1024 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Match Ultralytics 1024 -> 640 bilinear downscale
        rgb640 = cv2.resize(rgb1024, (640, 640), interpolation=cv2.INTER_LINEAR)

        bx, by, bw, bh = ann["bbox"]
        # Context window centered around box, clamped cleanly
        win = max(112, int(round(max(bw, bh) * 2.0)))
        win = min(win, 512)
        cx_box, cy_box = bx + bw * 0.5, by + bh * 0.5
        x0 = int(round(max(0, min(1024 - win, cx_box - win * 0.5))))
        y0 = int(round(max(0, min(1024 - win, cy_box - win * 0.5))))
        x1, y1 = x0 + win, y0 + win

        crop1024 = rgb1024[y0:y1, x0:x1]
        s = 640.0 / 1024.0
        sx0, sy0 = int(round(x0 * s)), int(round(y0 * s))
        sw = max(2, int(round(win * s)))
        sx1, sy1 = min(640, sx0 + sw), min(640, sy0 + sw)
        crop640 = rgb640[sy0:sy1, sx0:sx1]

        # Resize both crops with NEAREST to preserve true pixel structure side-by-side
        p_orig = Image.fromarray(crop1024).resize((cell_sz, cell_sz), Image.Resampling.NEAREST)
        p_640 = Image.fromarray(crop640).resize((cell_sz, cell_sz), Image.Resampling.NEAREST)

        def draw_outer_box(panel, ox, oy, ospan, bx_, by_, bw_, bh_, color):
            # Offset box outside by +3 display pixels so crack interior stays 100% visible
            sc = cell_sz / float(ospan)
            rx0 = (bx_ - ox) * sc - 3
            ry0 = (by_ - oy) * sc - 3
            rx1 = (bx_ + bw_ - ox) * sc + 3
            ry1 = (by_ + bh_ - oy) * sc + 3
            pd = ImageDraw.Draw(panel)
            pd.rectangle([rx0, ry0, rx1, ry1], outline=color, width=1)
            # Mark stitch line y=75% (768 on 1024, 480 on 640) if inside crop
            sy_stitch = (0.75 * (1024 if ospan == win else 640) - oy) * sc
            if 0 <= sy_stitch < cell_sz:
                pd.line([(0, sy_stitch), (cell_sz, sy_stitch)], fill=(255, 220, 40), width=1)
            # Label corner tag
            return panel

        box_col = (0, 230, 118) if is_ctrl else (255, 70, 70)
        p_orig = draw_outer_box(p_orig, x0, y0, win, bx, by, bw, bh, box_col)
        p_640 = draw_outer_box(p_640, sx0, sy0, sw, bx * s, by * s, bw * s, bh * s, box_col)

        # Header banner
        hdr_bg = (220, 245, 228) if is_ctrl else (253, 232, 232)
        draw.rectangle([cx, cy, cx + pair_w, cy + hdr_h - 2], fill=hdr_bg, outline=(180, 185, 190))
        tag = "PASS" if is_ctrl else ("4M-FAIL" if case["in_shared4"] else "2S-FAIL")
        line1 = f"[{tag}] {case['cls']} | id={case['ann_id']} | 640short={case['short640']:.1f}px ({bw:.0f}x{bh:.0f})"
        line2 = f"y=[{by/1024:.2f},{(by+bh)/1024:.2f}] | lvl=P{case['raw_best_level']} | iou42/100={case['iou42']:.2f}/{case['iou100']:.2f}"
        draw.text((cx + 4, cy + 2), line1, fill=(20, 20, 25), font=font)
        draw.text((cx + 4, cy + 16), line2, fill=(55, 60, 70), font=font)

        canvas.paste(p_orig, (cx, cy + hdr_h))
        canvas.paste(p_640, (cx + cell_sz + gap, cy + hdr_h))
        draw.rectangle([cx + 2, cy + hdr_h + 2, cx + 58, cy + hdr_h + 15], fill=(0, 0, 0))
        draw.text((cx + 5, cy + hdr_h + 4), "1024 orig", fill=(255, 255, 255), font=font)
        draw.rectangle([cx + cell_sz + gap + 2, cy + hdr_h + 2, cx + cell_sz + gap + 58, cy + hdr_h + 15], fill=(0, 0, 0))
        draw.text((cx + cell_sz + gap + 5, cy + hdr_h + 4), "640 scale", fill=(255, 255, 255), font=font)

    # Top title
    y_cur = margin
    draw.rectangle([margin, y_cur, canvas_w - margin, y_cur + section_h - 4], fill=(35, 45, 65))
    draw.text(
        (margin + 8, y_cur + 7),
        "SVRDD7 Val: 18 Representative Shared Narrow (<8px@640) raw_geometry@75 Failures (Left: 1024 orig | Right: 640 bilinear, outer GT box avoids occlusion)",
        fill=(255, 255, 255),
        font=font,
    )
    y_cur += section_h
    for idx, case in enumerate(fail_cases):
        r, c = divmod(idx, cols)
        render_pair(case, margin + c * (pair_w + 14), y_cur + r * (pair_h + 10), is_ctrl=False)

    y_cur += fail_rows * (pair_h + 10) + 8
    draw.rectangle([margin, y_cur, canvas_w - margin, y_cur + section_h - 4], fill=(28, 95, 58))
    draw.text(
        (margin + 8, y_cur + 7),
        "Control Group: 6 Narrow (<8px@640) Samples Retained@75 in BOTH CLS-P2A Seeds (same scale & classes)",
        fill=(255, 255, 255),
        font=font,
    )
    y_cur += section_h
    for idx, case in enumerate(ctrl_cases):
        r, c = divmod(idx, cols)
        render_pair(case, margin + c * (pair_w + 14), y_cur + r * (pair_h + 10), is_ctrl=True)

    canvas.save(out_path, optimize=True)


def main():
    train_stats = analyze_split("instances_train.json")
    val_stats = analyze_split("instances_val.json")

    gt_c42 = load_gt_map("HistoricalCLS_DualBest")
    gt_p42 = load_gt_map("HistCLS_SchemeA_P2")
    gt_c100 = load_gt_map("HistoricalCLS_seed100")
    gt_p100 = load_gt_map("HistCLS_SchemeA_P2_seed100")

    val_d = json.loads((ANN_DIR / "instances_val.json").read_text())
    val_anns = {int(a["id"]): a for a in val_d["annotations"]}
    val_imgs = {int(im["id"]): im for im in val_d["images"]}
    cats = {int(c["id"]): c["name"] for c in val_d["categories"]}

    fail_p42 = {aid for aid, r in gt_p42.items() if r["short640"] < 8.0 and r["stage75"] == "raw_geometry"}
    fail_p100 = {aid for aid, r in gt_p100.items() if r["short640"] < 8.0 and r["stage75"] == "raw_geometry"}
    fail_c42 = {aid for aid, r in gt_c42.items() if r["short640"] < 8.0 and r["stage75"] == "raw_geometry"}
    fail_c100 = {aid for aid, r in gt_c100.items() if r["short640"] < 8.0 and r["stage75"] == "raw_geometry"}
    shared2 = sorted(fail_p42 & fail_p100)
    shared4 = sorted(set(shared2) & fail_c42 & fail_c100)

    sh_cls = Counter()
    sh_lvl42 = Counter()
    sh_lvl100 = Counter()
    sh_zones = Counter()
    sh_orient = Counter()
    sh_iou50_both = 0
    sh_iou50_any = 0
    sh_shorts = []
    fail_records = []
    for aid in shared2:
        r42, r100 = gt_p42[aid], gt_p100[aid]
        ann = val_anns[aid]
        _, y, bw, bh = ann["bbox"]
        y1, y2 = y / 1024.0, (y + bh) / 1024.0
        sh_cls[r42["cls"]] += 1
        sh_lvl42[f"P{r42['raw_best_level']}"] += 1
        sh_lvl100[f"P{r100['raw_best_level']}"] += 1
        sh_zones[classify_zone(y1, y2)] += 1
        sh_shorts.append(r42["short640"])
        if bh < bw:
            sh_orient["h_lt_w"] += 1
        elif bw < bh:
            sh_orient["w_lt_h"] += 1
        else:
            sh_orient["eq"] += 1
        has50_42 = r42["raw_max_iou"] >= 0.50
        has50_100 = r100["raw_max_iou"] >= 0.50
        if has50_42 and has50_100:
            sh_iou50_both += 1
        if has50_42 or has50_100:
            sh_iou50_any += 1
        fail_records.append(
            dict(
                ann_id=aid,
                cls=r42["cls"],
                short640=round(r42["short640"], 2),
                bbox=[round(v, 1) for v in ann["bbox"]],
                in_shared4=(aid in shared4),
                raw_best_level=r42["raw_best_level"],
                iou42=round(r42["raw_max_iou"], 4),
                iou100=round(r100["raw_max_iou"], 4),
            )
        )

    # Pick 18 diverse representative fail cases (MC:4, TP:4, TC:4, LP:2, LC:2, P:2)
    quota = [("MC", 4), ("TP", 4), ("TC", 4), ("LP", 2), ("LC", 2), ("P", 2)]
    chosen_fail = []
    for cname, k in quota:
        sub = [x for x in fail_records if x["cls"] == cname]
        sub.sort(key=lambda x: (not x["in_shared4"], abs(x["iou42"] - 0.58), x["short640"]))
        # Spread across sorted positions to cover severe (<0.5) and near-miss (0.6-0.74)
        if len(sub) <= k:
            chosen_fail.extend(sub)
        else:
            sub_iou = sorted(sub, key=lambda x: (not x["in_shared4"], x["iou42"]))
            idxs = [int(round(i * (len(sub_iou) - 1) / (k - 1))) for i in range(k)]
            seen_i = set()
            for ix in idxs:
                while ix in seen_i and ix + 1 < len(sub_iou):
                    ix += 1
                seen_i.add(ix)
                chosen_fail.append(sub_iou[ix])

    # Pick 6 retained@75 control narrow cases (MC, TP, TC, LP, LC, P)
    ctrl_all = [
        dict(
            ann_id=aid,
            cls=gt_p42[aid]["cls"],
            short640=round(gt_p42[aid]["short640"], 2),
            bbox=[round(v, 1) for v in val_anns[aid]["bbox"]],
            in_shared4=False,
            raw_best_level=gt_p42[aid]["raw_best_level"],
            iou42=round(gt_p42[aid]["raw_max_iou"], 4),
            iou100=round(gt_p100[aid]["raw_max_iou"], 4),
        )
        for aid in sorted(gt_p42)
        if gt_p42[aid]["short640"] < 8.0
        and gt_p42[aid]["stage75"] == "retained"
        and gt_p100[aid]["stage75"] == "retained"
    ]
    chosen_ctrl = []
    for cname in ("MC", "TP", "TC", "LP", "LC", "P"):
        sub = [x for x in ctrl_all if x["cls"] == cname]
        sub.sort(key=lambda x: abs(x["short640"] - 5.5))
        if sub:
            chosen_ctrl.append(sub[0])

    val_img_map = {Path(ln.strip()).name: Path(ln.strip()) for ln in VAL_LIST.read_text().splitlines() if ln.strip()}
    collage_path = DIAG_DIR / "narrow102_visual_audit.png"
    make_visual_collage(val_img_map, val_anns, val_imgs, cats, chosen_fail, chosen_ctrl, collage_path)

    out_json = dict(
        train=train_stats,
        val=val_stats,
        shared_narrow_raw_geometry_at_75=dict(
            p2a_seed42_count=len(fail_p42),
            p2a_seed100_count=len(fail_p100),
            shared2_count=len(shared2),
            shared4_count=len(shared4),
            by_class=dict(sh_cls),
            by_best_level_seed42=dict(sh_lvl42),
            by_best_level_seed100=dict(sh_lvl100),
            by_zone=dict(sh_zones),
            orientation=dict(sh_orient),
            has_iou50_both_seeds=sh_iou50_both,
            has_iou50_either_seed=sh_iou50_any,
            short640_quantiles=quantiles(sh_shorts),
            visual_fail_samples=chosen_fail,
            visual_control_samples=chosen_ctrl,
        ),
    )
    json_path = DIAG_DIR / "svrdd_zones_and_102_audit.json"
    json_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False))
    print(f"AUDIT_OK: json={json_path} ({json_path.stat().st_size}B), png={collage_path} ({collage_path.stat().st_size}B)")
    print(f"shared2={len(shared2)}, shared4={len(shared4)}, iou50_both={sh_iou50_both}, cls={dict(sh_cls)}, lvl42={dict(sh_lvl42)}")


if __name__ == "__main__":
    main()
