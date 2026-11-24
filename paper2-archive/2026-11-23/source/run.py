"""Matched CLS@640 training and independent Val evaluation for the 42106 cohort."""
import argparse
import collections
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STAGE = Path(__file__).resolve().parent
ROOT = Path("/root/YOLO26N-FCHL")
REPORTS = ROOT / "reports"
GEO = REPORTS / "paper2_geometry_semantic_640_20261109_42266"
REF = REPORTS / "paper2_cls_p2p3_snake_640_20261115_42264"
BASE_INIT = GEO / "initialization/Cls_initial_seed42.pt"
BASE_SHA = "d4ceb0885be42e39be60b4b915024ccb420209ef6f4e2d7fc00edebec6947747"
STATE_SHA = "93fc618e80b0ccebfe2d45350264577cfd5413ce919782179f68fb9645a18101"
DATA_YAML = REF / "svrdd7.absolute.yaml"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path[:0] = [str(STAGE), str(GEO/"source"), str(REF/"source/dependencies"), str(GEO/"source/dependencies"), str(ROOT)]
import torch
import models  # Pickle identity of the untrained reference.
import experiment as reference
from coco_selection import COCOSelectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

sha, dump = reference.file_sha, reference.dump
torch.set_num_threads(2)

def now():
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()

def proof(arm):
    return STAGE/"records"/arm

def prepare(arm):
    assert sha(BASE_INIT) == BASE_SHA
    base = reference.load_model(BASE_INIT)
    assert reference.state_sha(base) == STATE_SHA and sum(p.numel() for p in base.parameters()) == 3195674
    head = base.model[-1]
    assert head.family == "cls" and head.geometry is None and not base.use_set
    assert tuple(head.stride.tolist()) == (4., 8., 16., 32.) and [len(s) for s in head.paper2_pki2.stages] == [2,2]
    assert base.model[1].conv.stride == (2,2)
    folder = proof(arm)
    folder.mkdir(exist_ok=True, parents=True)
    init = BASE_INIT
    model = base
    if arm != "C0":
        import candidates
        model, checks = candidates.prepare(base, arm)
        init = folder/"initial.pt"
        if init.exists():
            assert reference.state_sha(reference.load_model(init)) == reference.state_sha(model)
        else:
            torch.save(dict(model=model, ema=None, epoch=-1, optimizer=None), init)
        dump(folder/"candidate_checks.json", checks)
    config = dict(json.loads((REF/"protocol.json").read_text())["common_protocol"], model=str(init),
                  data=str(DATA_YAML), project=str(STAGE/"runs"), name=arm, exist_ok=False)
    assert (config["batch"],config["epochs"],config["imgsz"],config["seed"]) == (32,100,640,42)
    splits={}
    for name, expected in (("train",6000),("val",1000)):
        arr=(REF/"data"/f"{name}.txt").read_text().splitlines()
        assert len(arr)==len(set(arr))==expected and all("/test/" not in x.lower() for x in arr)
        splits[name]=set(Path(x).name for x in arr)
    assert splits["train"].isdisjoint(splits["val"])
    runtime_files = [Path(__file__), BASE_INIT, DATA_YAML, REF/"data/train.txt", REF/"data/val.txt",
                     reference.DATA/"annotations/instances_train.json", reference.DATA/"annotations/instances_val.json",
                     Path(models.__file__), Path(reference.__file__), Path(sys.modules["coco_selection"].__file__),
                     Path(sys.modules["rebuilt_model"].__file__), Path(sys.modules["pkinet_v2_author"].__file__)]
    if arm!="C0": runtime_files.append(Path(candidates.__file__))
    contract=dict(time=now(),arm=arm, initial=str(init), initial_sha256=sha(init), state_sha=reference.state_sha(model),
                  common_parent_sha256=BASE_SHA, params=sum(p.numel() for p in model.parameters()),
                  hashes={str(p):sha(p) for p in runtime_files}, protocol=config)
    dump(folder/"contract.json",contract)
    return contract

def first_batch(batch):
    receipt = {k:hashlib.sha256(batch[k].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
               for k in ("img","cls","bboxes","batch_idx")}
    receipt.update(names=[Path(p).name for p in batch["im_file"]], shape=list(batch["img"].shape))
    return receipt

def train(arm):
    if (STAGE/"runs"/arm).exists():
        raise FileExistsError(f"Existing run: {arm}")
    c=prepare(arm)
    folder=proof(arm)
    expected = collections.Counter(Path(x).name for x in (REF/"data/train.txt").read_text().splitlines())
    class Trainer(COCOSelectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            return reference.load_model(c["initial"])
        def preprocess_batch(self,batch):
            self.exposure.update(Path(p).name for p in batch["im_file"])
            result=super().preprocess_batch(batch)
            if not hasattr(self,"first_receipt"):
                self.first_receipt=first_batch(result)
                dump(folder/"first_batch.json",self.first_receipt)
                if arm!="C0":
                    assert self.first_receipt == json.loads((proof("C0")/"first_batch.json").read_text())
            return result
        def optimizer_step(self):
            before=self.scaler.get_scale()
            super().optimizer_step()
            if self.scaler.get_scale()<before: self.skips+=1
            else: self.success+=1
    def started(t):
        m=unwrap_model(t.model)
        assert reference.state_sha(m)==c["state_sha"]
        assert len(t.train_loader.dataset)==6000 and len(t.test_loader.dataset)==1000
        assert sum(len(row["cls"]) for row in t.train_loader.dataset.labels)==15619
        assert sum(len(row["cls"]) for row in t.test_loader.dataset.labels)==2612
        assert type(t.optimizer).__name__=="MuSGD" and t.amp and t.batch_size==32 and t.epochs==100
        assert t.args.nbs==64 and t.accumulate==2
        t.success=t.skips=0
        criterion=m.init_criterion()
        assert all(branch.assigner.stride==[8.,16.,32.] and branch.reg_max==1
                   for branch in (criterion.one2many,criterion.one2one))
        m.criterion=criterion
        torch.cuda.reset_peak_memory_stats()
        t.peak_alloc=t.peak_res=0
        dump(folder/"runtime.json",dict(time=now(),pid=os.getpid(),gpu=torch.cuda.get_device_name(0),
            state_sha=c["state_sha"], params=c["params"], optimizer=type(t.optimizer).__name__, training=True))
    def epoch_start(t):
        t.exposure=collections.Counter()
        t.epoch_wall=time.monotonic()
    def record(t):
        if not t.validator.training: return
        assert t.exposure==expected and unwrap_model(t.model).criterion.updates==t.epoch+1
        t.peak_alloc=max(t.peak_alloc,torch.cuda.max_memory_allocated())
        t.peak_res=max(t.peak_res,torch.cuda.max_memory_reserved())
        row=dict(time=now(), epoch=t.epoch+1, primary=sum(t.exposure.values()),seconds=time.monotonic()-t.epoch_wall,
                 successful_steps=t.success,amp_skips=t.skips,metrics=t.metrics,peak_allocated_GiB=t.peak_alloc/2**30,
                 peak_reserved_GiB=t.peak_res/2**30)
        with (folder/"epochs.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        dump(folder/"status.json",row)
    t=Trainer(overrides=c["protocol"])
    for event,fn in (("on_train_start",started),("on_train_epoch_start",epoch_start),("on_fit_epoch_end",record)):
        t.add_callback(event,fn)
    t.train()
    rows=[json.loads(x) for x in (folder/"epochs.jsonl").read_text().splitlines()]
    assert [x["epoch"] for x in rows]==list(range(1,101))
    assert all(sha(Path(path))==h for path,h in c["hashes"].items())
    dump(folder/"trained.json", dict(time=now(), epochs=100,primary=sum(x["primary"] for x in rows),state_sha=c["state_sha"]))

def evaluate(arm):
    if arm != "C0":
        import candidates
    folder=proof(arm)
    assert (folder/"trained.json").is_file()
    result={}
    for kind in ["best","last"]:
        ck=STAGE/"runs"/arm/"weights"/f"{kind}.pt"
        path=folder/"evaluation"/kind
        if (path/"metrics.json").exists():
            r=json.loads((path/"metrics.json").read_text()); assert r["sha256"]==sha(ck)
        else:
            if path.exists(): path=path.with_name(kind+"_"+str(time.time_ns()))
            r=reference.evaluate_checkpoint(ck,path,DATA_YAML)
        result[kind]=r
    dump(folder/"evaluation/summary.json",result)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("train","evaluate","prepare"))
    parser.add_argument("arm")
    args=parser.parse_args()
    {"train":train,"evaluate":evaluate,"prepare":prepare}[args.phase](args.arm)
