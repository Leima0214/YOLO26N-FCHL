"""Replay the frozen October CLS initialization with native checkpoint selection."""
import collections
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

import run
import torch
from historical_head import bind
from recover_historical_initial import OLD_ALL, OLD_P2, OLD_PKI, digests
from ultralytics.utils.torch_utils import unwrap_model

ARM = "HistoricalCLS"
FOLDER = run.proof(ARM)
RUN = run.STAGE / "runs" / ARM
INITIAL = run.STAGE / "identity_audit/historical_replay/CLS640_historical_initial.pt"
EXPECTED = dict(all=OLD_ALL, P2=OLD_P2, PKI=OLD_PKI)
ARCHIVE = Path("/SVRDD/SVRDD_YOLO_READY/_paper2_archives") / run.STAGE.name / "historical_replay"
OLD_PIXELS = "92208b9a827c4a23397785d8a8254f820e456d1abefecbfa4a21c5d85a5eeb43"


def read(path):
    return json.loads(path.read_text())


def prepare():
    assert run.sha(INITIAL) == "71da1a0dd162ec45903f6e146c5852b68c70cfa52bd596de667ffeabae4bfc29"
    model = bind(run.reference.load_model(INITIAL))
    assert digests(model) == EXPECTED and sum(p.numel() for p in model.parameters()) == 3195674
    head = model.model[-1]
    bn = [(x.eps, x.momentum) for x in head.p2_fusion.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    assert bn == [(.001, .03)] * 4 and head.stride.tolist() == [4, 8, 16, 32]
    control = run.reference.load_model(INITIAL).model[-1].eval()
    head.eval()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(42)
        features = [torch.randn(2,c,h,h) for c,h in [(64,32),(64,16),(128,8),(256,4)]]
        for h in (control, head):
            h.paper2_pki2.alpha.copy_(torch.tensor([.1,-.1]))
        before, after = control(features)[1], head(features)[1]
        for branch in ("one2many", "one2one"):
            for k in ("scores","boxes"):
                assert torch.equal(before[branch][k], after[branch][k]), (branch, k)
    cfg = dict(read(run.REF / "protocol.json")["common_protocol"], model=str(INITIAL),
               data=str(run.DATA_YAML), project=str(RUN.parent), name=ARM, exist_ok=False, save_json=False)
    assert (cfg["imgsz"],cfg["epochs"],cfg["batch"],cfg["seed"],cfg["optimizer"]) == (640,100,32,42,"MuSGD")
    FOLDER.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), run.STAGE/"historical_head.py", Path(run.reference.__file__),
               Path(run.models.__file__), Path(sys.modules["rebuilt_model"].__file__)]
    run.dump(FOLDER/"contract.json", dict(time=run.now(),initial_file=run.sha(INITIAL),state=EXPECTED,
             params=3195674,p2_bn=bn,old_pixel_hash=OLD_PIXELS,config=cfg,
             selection="Native validation fitness; independent FP32 COCO Val for best and last",
             head_comparison="Identical O2M/O2O scores and boxes with nonzero gates on CPU",
             sources={str(p):run.sha(p) for p in sources}))
    return cfg


def train(cfg):
    if RUN.exists():
        raise FileExistsError(f"Existing run needs inspection: {RUN}")
    expected = collections.Counter(Path(x).name for x in (run.REF/"data/train.txt").read_text().splitlines())

    class Trainer(run.reference.TaskTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return bind(run.reference.load_model(INITIAL))

        def preprocess_batch(self, batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result = super().preprocess_batch(batch)
            path = FOLDER/"first_batch.json"
            if not path.exists():
                receipt = run.first_batch(result)
                c0 = read(run.proof("C0")/"first_batch.json")
                run.dump(path, dict(**receipt, same_as_C0={k:receipt[k]==v for k,v in c0.items()},
                                   same_pixels_as_october=receipt["img"]==OLD_PIXELS))
            return result

    def started(t):
        model = unwrap_model(t.model)
        assert digests(model) == EXPECTED
        assert (len(t.train_loader.dataset),len(t.test_loader.dataset)) == (6000,1000)
        assert type(t.optimizer).__name__ == "MuSGD" and t.amp and t.batch_size == 32
        assert t.accumulate==2 and not t.args.save_json
        criterion = model.init_criterion()
        assert all(b.assigner.stride == [8.,16.,32.] for b in (criterion.one2one,criterion.one2many))
        model.criterion = criterion
        run.dump(FOLDER/"runtime.json",dict(time=run.now(),pid=os.getpid(),gpu=torch.cuda.get_device_name(),
            state=EXPECTED,train_batch=t.batch_size,val_batch=t.test_loader.batch_size,optimizer=type(t.optimizer).__name__))

    def epoch_start(t):
        t.exposure = collections.Counter()

    def record(t):
        if not t.validator.training:
            return
        assert t.exposure == expected and unwrap_model(t.model).criterion.updates == t.epoch+1
        row = dict(time=run.now(),epoch=t.epoch+1,images=sum(t.exposure.values()),metrics=t.metrics)
        with (FOLDER/"epochs.jsonl").open("a") as out:
            out.write(json.dumps(row)+"\n")
        run.dump(FOLDER/"status.json",row)

    t = Trainer(overrides=cfg)
    for event, cb in [("on_train_start",started),("on_train_epoch_start",epoch_start),("on_fit_epoch_end",record)]:
        t.add_callback(event,cb)
    t.train()
    rows = [json.loads(line) for line in (FOLDER/"epochs.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in rows] == list(range(1,101))
    epochs = read(RUN/"checkpoint_epoch_map.json")
    assert 1 <= epochs["best"] <= 100 and epochs["last"] == 100
    run.dump(FOLDER/"trained.json",dict(time=run.now(),epochs=epochs,
             late_native_val={k:statistics.mean(row["metrics"][k] for row in rows[-10:]) for k in
                              ["metrics/mAP50(B)","metrics/AP75(B)","metrics/mAP50-95(B)"]}))


def archive(src, rel):
    dest = ARCHIVE/rel
    dest.parent.mkdir(exist_ok=True,parents=True)
    if dest.is_file() and run.sha(src)==run.sha(dest):
        return
    tmp=dest.with_suffix(dest.suffix+".tmp")
    shutil.copy2(src,tmp)
    assert run.sha(src)==run.sha(tmp)
    tmp.replace(dest)


if __name__ == "__main__":
    if sys.argv[1:] not in (["prepare"], ["all"]):
        raise SystemExit("usage: historical_replay.py prepare|all")
    cfg=prepare()
    print("October initial state and both forward branches verified", OLD_ALL, flush=True)
    if sys.argv[1]=="all":
        if not (FOLDER/"trained.json").exists():
            train(cfg)
        for src in [RUN/"weights/best.pt",RUN/"weights/last.pt",RUN/"args.yaml",RUN/"results.csv",RUN/"checkpoint_epoch_map.json"]:
            archive(src,Path("runs")/ARM/src.relative_to(RUN))
        for src in [run.STAGE/"historical_head.py",Path(__file__), run.STAGE/"recover_historical_initial.py"]:
            archive(src,Path("source")/src.name)
        run.evaluate(ARM)
        for src in FOLDER.rglob("*"):
            if src.is_file():
                archive(src,Path("records")/ARM/src.relative_to(FOLDER))
        print("Historical CLS training and independent best/last Val complete", flush=True)
