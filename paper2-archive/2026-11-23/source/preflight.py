"""Check branch routing and learning from zero initialization using the real head's feature shapes."""
import torch
from run import STAGE, now, dump, reference
from candidates import ARMS

def head_gradients(model, arm):
    head = model.model[-1]
    raw = []
    def capture(_module, batch):
        raw.extend([v.detach().requires_grad_(True) for v in batch[0]])
    handle = head.register_forward_pre_hook(capture)
    with torch.no_grad():
        model(torch.randn(2,3,128,160))
    handle.remove()
    result = head(raw)
    keys = ("ctx_o2m","ctx_o2o") if arm=="A" else ("gather_out",)
    checks = {}
    for branch in ("one2one","one2many"):
        height,width = raw[0].shape[-2:]
        result[branch]["scores"][:,:,:height*width].mean().backward(retain_graph=True)
        grad = {}
        for name in keys:
            g=getattr(head,name).weight.grad
            grad[name]=float(g.abs().sum()) if g is not None else 0.
        grad["inputs"]=sum(float(v.grad.abs().sum()) for v in raw if v.grad is not None)
        checks[branch]=grad
        head.zero_grad(set_to_none=True)
        for v in raw: v.grad=None
    o,m=checks["one2one"],checks["one2many"]
    assert o["inputs"]==0 and m["inputs"]>0
    if arm=="A": assert o["ctx_o2o"]>0 and o["ctx_o2m"]==0 and m["ctx_o2m"]>0 and m["ctx_o2o"]==0
    else: assert o["gather_out"]==0 and m["gather_out"]>0
    return checks

def haar_gradients(model, arm):
    layer=model.model[1]
    x=torch.randn(2,layer.conv.in_channels,33,46)
    target=torch.randn_like(layer(x)).detach()
    steps=[]
    for _ in range(2):
        loss=(layer(x)-target).square().mean()
        loss.backward()
        steps.append({k:float(v.grad.abs().sum()) if v.grad is not None else 0.
                      for k,v in layer.named_parameters() if k in ("gamma","pw.weight","gate.0.weight")})
        with torch.no_grad():
            for v in layer.parameters():
                if v.grad is not None: v.add_(v.grad,alpha=-.01)
        layer.zero_grad(set_to_none=True)
    assert steps[0]["gamma"]>0 and steps[0]["pw.weight"]==0 and steps[1]["pw.weight"]>0
    if arm.startswith("Dual"): assert steps[1]["gate.0.weight"]>0
    return steps

if __name__=="__main__":
    torch.manual_seed(57)
    torch.set_num_threads(1)
    results={}
    for arm in ARMS:
        model=reference.load_model(STAGE/"records"/arm/"initial.pt").train()
        results[arm]=head_gradients(model,arm) if arm=="A" or arm.startswith("G1_") else haar_gradients(model,arm)
    dump(STAGE/"candidate_gradient_checks.json",dict(time=now(),checks=results,
         scope="CPU synthetic gradient and routing checks at the saved initial states"))
    print("Passed:",*results)

