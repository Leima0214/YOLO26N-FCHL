"""Recover the archived October CLS starting tensors on CPU. This never starts training."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from run import BASE_INIT, STAGE, ROOT, reference, sha, dump, now
import rebuilt_model as plain
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import init_seeds

OLD_P2="b1d082f03692fb8594beca8cdcba1c727ed77b5ee31020348e90f9181385c452"
OLD_PKI="2b2180792165da779dbbeb521c280c7882c368bc4acdaf6c077d220595f9c65a"
OLD_ALL="26a0846197478420906d3485a3a7e35d4d2ed84643c54a012aaa2bc67d9c4c3f"

def historical_hash(state):
    h=hashlib.sha256()
    for key,t in sorted(state.items()):
        h.update(key.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def digests(model):
    state={k.replace(".p2_fuse.",".p2_fusion."):v for k,v in model.state_dict().items()}
    return dict(all=historical_hash(state),P2=historical_hash({k:v for k,v in state.items() if ".paper2_pki2." not in k}),
                PKI=historical_hash(model.model[-1].paper2_pki2.state_dict()))

def build():
    assert sha(plain.PUBLIC)==plain.PUBLIC_SHA
    # Original native7 constructs the seven-class model before loading public COCO weights.
    init_seeds(42,deterministic=True)
    cfg=YAML.load(ROOT/"ultralytics/cfg/models/26/yolo26.yaml")
    cfg["scale"]="n"
    native=DetectionModel(cfg,nc=7,verbose=False).float()
    native.load(YOLO(str(plain.PUBLIC),task="detect").model,verbose=False)
    model=reference.load_model(BASE_INIT)
    orig=model.model[-1]
    # The old builder seeded the P2 head at its own installation step.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        head=plain.P2TaskDetect()
        head.bias_init()
    for group in ["cv2","cv3","one2one_cv2","one2one_cv3"]:
        for i,branch in enumerate(getattr(native.model[-1],group)):
            getattr(head,group)[i+1].load_state_dict(branch.state_dict(),strict=True)
    model.model[:-1].load_state_dict(native.model[:-1].state_dict(),strict=True)
    orig.load_state_dict(head.state_dict(),strict=True)
    for bn in orig.p2_fuse.modules():
        if isinstance(bn,torch.nn.BatchNorm2d):
            bn.eps,bn.momentum=.001,.03
    return model

if __name__=="__main__":
    torch.set_num_threads(1)
    path=STAGE/"identity_audit"/"historical_replay"
    path.mkdir(parents=True,exist_ok=True)
    candidate=build()
    observed=digests(candidate)
    target=dict(all=OLD_ALL,P2=OLD_P2,PKI=OLD_PKI)
    receipt=dict(time=now(),historical=target,observed=observed,matched=observed==target,
                 C0=digests(reference.load_model(BASE_INIT)),public_sha256=sha(plain.PUBLIC),
                 source_sha256=sha(Path(__file__)),historical_P2_BN=[.001,.03],training_started=False)
    if receipt["matched"]:
        ck=path/"CLS640_historical_initial.pt"
        if not ck.exists():
            tmp=ck.with_suffix(".tmp")
            torch.save(dict(epoch=-1,model=candidate,ema=None,optimizer=None),tmp)
            tmp.replace(ck)
        loaded=reference.load_model(ck)
        assert digests(loaded)==target
        bn=[(v.eps,v.momentum) for v in loaded.model[-1].p2_fuse.modules() if isinstance(v,torch.nn.BatchNorm2d)]
        assert len(bn)==4 and all(v==(.001,.03) for v in bn)
        receipt.update(checkpoint=str(ck),file_sha256=sha(ck),bn_count=len(bn))
    dump(path/"initialization_audit.json",receipt)
    print(json.dumps(receipt))
