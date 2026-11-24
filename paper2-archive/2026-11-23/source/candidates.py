"""Single-site candidates built from exactly the saved C0 state."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from ultralytics.nn.modules import Conv, DWConv, Detect
from models import FusedDetect

ARMS = ("A", "H1_L1", "Dual_L1", "Dual_L1_static", "G1_P2_40", "G1_P2_80")
ADDED = dict(A=12376, H1_L1=2113, Dual_L1=3232, Dual_L1_static=3232, G1_P2_40=51072, G1_P2_80=51072)

def inputs(head, raw):
    c2,p3,p4,p5=raw
    p2=head.p2_fuse(torch.cat((c2,F.interpolate(p3,size=c2.shape[-2:],mode="nearest")),1))
    q4,q5=head.paper2_pki2(p4,p5)
    return [p2,p3,p4,p5,p2,p3,q4,q5]

class ScoreContext(FusedDetect):
    def forward(self, raw):
        streams=inputs(self,raw)
        p2,p3=streams[:2]
        context=self.context(torch.cat((F.pixel_unshuffle(p2,2),p3),1))
        return Detect.forward(self,[*streams,context])

    def forward_head(self,x,box_head=None,cls_head=None):
        result=super().forward_head(x[:8],box_head,cls_head)
        if not result: return result
        projection=self.ctx_o2m if cls_head is self.cv3 else self.ctx_o2o
        delta=F.pixel_shuffle(projection(x[8]),2).flatten(2)
        original=result["scores"]
        n=delta.shape[-1]
        result["scores"]=torch.cat((original[:,:,:n]+delta,original[:,:,n:]),2)
        return result

class LocalGather(FusedDetect):
    def forward(self,raw):
        streams=inputs(self,raw)
        target=streams[self.gather_level].shape[-2:]
        aligned=[]
        for t in streams[:4]:
            if t.shape[-2:]==target: aligned.append(t)
            elif t.shape[-2]>=target[0] and t.shape[-1]>=target[1]:
                aligned.append(F.adaptive_avg_pool2d(t,target))
            else: aligned.append(F.interpolate(t,size=target,mode="bilinear",align_corners=False))
        g=self.gather_in(torch.cat(aligned,1))
        for block in self.gather_blocks:g=g+block(g)
        p2=streams[0]
        a,e=F.interpolate(self.gather_out(g),size=p2.shape[-2:],mode="bilinear",align_corners=False).chunk(2,1)
        streams[4]=p2*(2*torch.sigmoid(a.float())).to(p2.dtype)+e
        return Detect.forward(self,streams)

def haar(x):
    if x.shape[-1]%2 or x.shape[-2]%2:
        x=F.pad(x,(0,x.shape[-1]%2,0,x.shape[-2]%2))
    xf=x.float() if x.dtype in (torch.float16,torch.bfloat16) else x
    a,b,c,d=xf[...,::2,::2],xf[...,::2,1::2],xf[...,1::2,::2],xf[...,1::2,1::2]
    return tuple((v*0.5).to(x.dtype) for v in (a+b+c+d,a+b-c-d,a-b+c-d,a-b-c+d))

class HaarResidual(nn.Module):
    def __init__(self,orig,kind):
        super().__init__()
        self.kind=kind
        self.conv,self.bn,self.act=orig.conv,orig.bn,orig.act
        ci,co=self.conv.in_channels,self.conv.out_channels
        assert self.conv.kernel_size==(3,3) and self.conv.stride==(2,2) and self.conv.padding==(1,1)
        n=4*ci if kind=="H1_L1" else 3*ci
        self.pw=nn.Conv2d(n,co,1,bias=False)
        self.hbn=nn.BatchNorm2d(co,eps=self.bn.eps,momentum=self.bn.momentum)
        self.gamma=nn.Parameter(torch.zeros(1) if kind=="H1_L1" else torch.zeros(1,co,1,1))
        if kind!="H1_L1":
            mid=max(16,n//4)
            self.gate=nn.Sequential(nn.Conv2d(n,mid,1),nn.SiLU(),nn.Conv2d(mid,n,1))
        self.i,self.f,self.type=orig.i,orig.f,kind
        self.np=sum(p.numel() for p in self.parameters())

    def forward(self,x):
        bands=haar(x)
        if self.kind=="H1_L1":
            y=torch.cat(bands,1)
        else:
            y=torch.cat(bands[1:],1)
            pooled=y.mean((2,3),keepdim=True) if self.kind=="Dual_L1" else torch.ones_like(y[:,:,:1,:1])
            y=y*torch.sigmoid(self.gate(pooled).float()).to(y.dtype)
        return self.act(self.bn(self.conv(x))+self.gamma.to(x.dtype)*self.hbn(self.pw(y)))

    @torch.no_grad()
    def folded(self):
        assert self.kind=="H1_L1" and not self.training
        base=fuse_conv_bn_eval(self.conv,self.bn)
        proj=fuse_conv_bn_eval(self.pw,self.hbn)
        co,ci=base.weight.shape[:2]
        H=base.weight.new_tensor([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])*0.5
        k=torch.einsum("obc,bs->ocs",proj.weight[:,:,0,0].reshape(co,4,ci),H).reshape(co,ci,2,2)
        base.weight[:,:,1:,1:].add_(self.gamma*k)
        base.bias.add_(self.gamma*proj.bias)
        out=Conv(ci,co,3,2)
        out.conv,out.act=base,copy.deepcopy(self.act)
        del out.bn
        out.forward=out.forward_fuse
        out.i,out.f,out.type=self.i,self.f,"folded H1 Conv"
        out.np=sum(p.numel() for p in out.parameters())
        return out.eval()

def build(parent,arm):
    if arm not in ARMS: raise ValueError(arm)
    model=copy.deepcopy(parent)
    head=model.model[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if arm=="A":
            head.__class__=ScoreContext
            head.context=nn.Sequential(nn.Conv2d(320,32,1,bias=False),nn.SiLU(),
                                       nn.Conv2d(32,32,3,padding=1,groups=32,bias=False),nn.SiLU())
            head.ctx_o2m=nn.Conv2d(32,28,1)
            head.ctx_o2o=nn.Conv2d(32,28,1)
            for m in (head.ctx_o2m,head.ctx_o2o):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        elif arm.startswith("G1_"):
            head.__class__=LocalGather
            head.gather_level=2 if arm.endswith("40") else 1
            head.gather_in=Conv(512,64,1)
            head.gather_blocks=nn.ModuleList([nn.Sequential(DWConv(64,64,3),Conv(64,64,1)) for _ in range(2)])
            head.gather_out=nn.Conv2d(64,128,1)
            nn.init.zeros_(head.gather_out.weight);nn.init.zeros_(head.gather_out.bias)
        else: model.model[1]=HaarResidual(model.model[1],arm)
    return model

@torch.no_grad()
def prepare(parent,arm):
    candidate=build(parent,arm)
    old=parent.state_dict(); new=candidate.state_dict()
    assert all(k in new and torch.equal(v,new[k]) for k,v in old.items())
    n=sum(p.numel() for p in candidate.parameters())
    assert n==3195674+ADDED[arm]
    # Use copies to keep BN buffers, anchor caches and model mode pristine in the saved initial state.
    a,b=copy.deepcopy(parent).eval(),copy.deepcopy(candidate).eval()
    x=torch.randn(1,3,128,160,generator=torch.Generator().manual_seed(43))
    ra,rb=a(x)[1],b(x)[1]
    checks={branch:{k:float((ra[branch][k]-rb[branch][k]).abs().max()) for k in ("boxes","scores")}
            for branch in ("one2many","one2one")}
    assert all(val==0 for metrics in checks.values() for val in metrics.values()),checks
    if arm=="H1_L1":
        h=b.model[1]; h.gamma.fill_(0.37)
        u=torch.randn(2,16,17,26,generator=torch.Generator().manual_seed(42))
        error=float((h(u)-h.folded()(u)).abs().max()); assert error<1e-4
        checks["folded_max_error"]=error
    return candidate,dict(arm=arm,params=n,added=ADDED[arm],inherited_tensors=len(old),initial_predictions=checks)

def compact_o2m(model):
    from rebuilt_model import as_o2m
    out=as_o2m(model)
    head=out.model[-1]
    if isinstance(head,ScoreContext) and hasattr(head,"ctx_o2o"): del head.ctx_o2o
    if isinstance(out.model[1],HaarResidual) and out.model[1].kind=="H1_L1":
        out.model[1]=out.model[1].folded()
    return out
