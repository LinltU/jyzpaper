"""Independent paper-inspired forecasting branches for the insulator data.

The upstream repositories are kept under ``../upstream``.  Their public data
loaders target very different benchmarks, therefore this module implements the
algorithmic cores against the causal 24-step local windows.  The branches are
deliberately separate classes and training loops:

* TimeGrad: GRU state + autoregressive conditional DDPM.
* DiffPLF: trend/scale decomposition + residual diffusion.
* CCDM: channel-specific encoders and channel dropout/consistency.
* TSDE: mask-conditioned GRU trained with forecast/interpolation/imputation
  masks (IIF-style self supervision).
* NsDiff: per-window non-stationary normalization and FiLM conditioning.

All branches predict ``target - persistence`` in standardized target units and
return samples with shape [n_samples, n_rows].
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    z = t.float().unsqueeze(1) * freq.unsqueeze(0)
    out = torch.cat([torch.sin(z), torch.cos(z)], dim=1)
    return nn.functional.pad(out, (0, max(0, dim - out.shape[1])))[:, :dim]


class DiffusionSchedule:
    def __init__(self, steps: int):
        self.steps = int(steps)
        self.beta = torch.linspace(1e-4, 0.02, self.steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)


class _ScoreHead(nn.Module):
    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + 1 + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, z, t, cond):
        return self.net(torch.cat([z[:, None], time_embedding(t, 32), cond], 1)).squeeze(1)


class _BaseBranch:
    """Shared numerical DDPM utilities; subclasses own encoder/loss logic."""
    name = "base"

    def _diffusion_fit(self, residual, cond, score, epochs, batch_size, lr, seed, schedule, extra_loss=None):
        seed_everything(seed)
        device = torch.device("cpu")
        residual = torch.from_numpy(residual).float()
        cond = torch.from_numpy(cond).float()
        scale = float(residual.std().clamp_min(0.05))
        loader = DataLoader(TensorDataset(residual, cond), batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(score.parameters(), lr=lr, weight_decay=2e-4)
        score.train(); history = []
        for epoch in range(1, epochs + 1):
            losses = []
            for r, c in loader:
                r = r / scale
                t = torch.randint(0, schedule.steps, (len(r),))
                ab = schedule.alpha_bar[t]
                noise = torch.randn_like(r)
                z = ab.sqrt() * r + (1 - ab).sqrt() * noise
                pred = score(z, t.float() / max(schedule.steps - 1, 1), c)
                loss = (pred - noise).pow(2).mean()
                if extra_loss is not None:
                    loss = loss + extra_loss(r, pred, c)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(score.parameters(), 1.0); opt.step()
                losses.append(float(loss.detach()))
            history.append({"epoch": epoch, "loss": float(np.mean(losses)), "residual_scale": scale})
        return scale, history

    @torch.no_grad()
    def _diffusion_sample(self, score, cond, base, scale, schedule, n_samples, seed):
        score.eval(); cond = torch.from_numpy(cond).float(); base = torch.from_numpy(base).float()
        outputs = []
        for sample_id in range(n_samples):
            g = torch.Generator().manual_seed(seed + sample_id)
            z = torch.randn(len(cond), generator=g)
            for k in range(schedule.steps - 1, -1, -1):
                tt = torch.full((len(cond),), k / max(schedule.steps - 1, 1))
                eps = score(z, tt, cond)
                a, ab = schedule.alpha[k], schedule.alpha_bar[k]
                mean = (z - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
                if k > 0: mean = mean + schedule.beta[k].sqrt() * torch.randn(len(cond), generator=g)
                z = mean
            outputs.append((base + scale * z).numpy())
        return np.stack(outputs)


class TimeGradBranch(_BaseBranch):
    """TimeGrad core: recurrent hidden state conditions each diffusion step."""
    name = "TimeGrad"
    def __init__(self, history_dim, context_dim, hidden=64, steps=8, seed=42):
        self.seed = seed; self.schedule = DiffusionSchedule(steps)
        self.gru = nn.GRU(history_dim, hidden, batch_first=True)
        self.context = nn.Sequential(nn.Linear(context_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.score = _ScoreHead(hidden * 2, hidden)
    def _condition(self, x, c):
        h, _ = self.gru(x); return torch.cat([h[:, -1], self.context(c)], 1)
    def fit(self, x, c, target, persistence, epochs=6, batch_size=512, lr=8e-4):
        seed_everything(self.seed); xt, ct = torch.from_numpy(x).float(), torch.from_numpy(c).float()
        base = torch.from_numpy(persistence).float(); residual = torch.from_numpy(target - persistence).float()
        ds = DataLoader(TensorDataset(xt, ct, residual), batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(list(self.gru.parameters()) + list(self.context.parameters()) + list(self.score.parameters()), lr=lr)
        scale = float(residual.std().clamp_min(0.05)); self._scale = scale; hist=[]
        for ep in range(1, epochs+1):
            ls=[]; self.gru.train(); self.context.train(); self.score.train()
            for xb, cb, rb in ds:
                t=torch.randint(0,self.schedule.steps,(len(xb),)); ab=self.schedule.alpha_bar[t]; n=torch.randn_like(rb); z=ab.sqrt()*(rb/scale)+(1-ab).sqrt()*n
                pred=self.score(z,t.float()/max(self.schedule.steps-1,1),self._condition(xb,cb)); loss=(pred-n).pow(2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(self.parameters()),1.0); opt.step(); ls.append(float(loss.detach()))
            hist.append({"epoch":ep,"loss":float(np.mean(ls)),"residual_scale":scale})
        return hist
    def parameters(self): return list(self.gru.parameters())+list(self.context.parameters())+list(self.score.parameters())
    @torch.no_grad()
    def sample(self,x,c,persistence,n_samples=16):
        self.gru.eval(); self.context.eval(); self.score.eval(); cond=self._condition(torch.from_numpy(x).float(),torch.from_numpy(c).float()).numpy()
        return self._diffusion_sample(self.score,cond,persistence,self._scale,self.schedule,n_samples,self.seed)


class DiffPLFBranch(_BaseBranch):
    """DiffPLF core: explicit local linear forecast and locally scaled residual."""
    name = "DiffPLF"
    def __init__(self, history_dim, context_dim, hidden=64, steps=8, seed=42):
        self.seed=seed; self.schedule=DiffusionSchedule(steps)
        self.encoder=nn.Sequential(nn.Linear(history_dim*6+context_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.score=_ScoreHead(hidden,hidden)
    def _features(self,x,c):
        tail=x[:,-6:,:].reshape(len(x),-1); return torch.cat([tail,c],1)
    def _base_scale(self,x,target,persistence):
        m=x[:,:,0]; slope=(m[:,-1]-m[:,0])/5.0; base=m[:,-1]+torch.clamp(slope,-0.35,0.35)
        local=(m-m.mean(1,keepdim=True)).abs().mean(1).clamp_min(.05)
        return base,local
    def fit(self,x,c,target,persistence,epochs=6,batch_size=512,lr=8e-4):
        seed_everything(self.seed); xt,ct,yt=torch.from_numpy(x).float(),torch.from_numpy(c).float(),torch.from_numpy(target).float()
        base,local=self._base_scale(xt,yt,torch.from_numpy(persistence).float()); r=yt-base; self._local_train=float(local.mean())
        ds=DataLoader(TensorDataset(xt,ct,r,local),batch_size=batch_size,shuffle=True); opt=torch.optim.AdamW(list(self.encoder.parameters())+list(self.score.parameters()),lr=lr)
        self._scale=float(r.std().clamp_min(.05)); hist=[]
        for ep in range(1,epochs+1):
            ls=[]
            for xb,cb,rb,sb in ds:
                t=torch.randint(0,self.schedule.steps,(len(xb),)); ab=self.schedule.alpha_bar[t]; n=torch.randn_like(rb); z=ab.sqrt()*(rb/self._scale)+(1-ab).sqrt()*n
                cond=self.encoder(self._features(xb,cb)); pred=self.score(z,t.float()/max(self.schedule.steps-1,1),cond)
                loss=(pred-n).pow(2).mean()+0.03*(rb.abs()/sb).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.score.parameters()),1.); opt.step(); ls.append(float(loss.detach()))
            hist.append({"epoch":ep,"loss":float(np.mean(ls)),"residual_scale":self._scale})
        return hist
    @torch.no_grad()
    def sample(self,x,c,persistence,n_samples=16):
        xt=torch.from_numpy(x).float(); cond=self.encoder(self._features(xt,torch.from_numpy(c).float())).numpy(); base,_=self._base_scale(xt,None,torch.from_numpy(persistence).float()); return self._diffusion_sample(self.score,cond,base.numpy(),self._scale,self.schedule,n_samples,self.seed)


class CCDMBranch(_BaseBranch):
    """CCDM core: independent channel projections plus cross-channel fusion."""
    name="CCDM"
    def __init__(self,history_dim,context_dim,hidden=64,steps=8,seed=42):
        self.seed=seed; self.schedule=DiffusionSchedule(steps)
        self.channel_proj=nn.ModuleList([nn.Sequential(nn.Linear(1,hidden//2),nn.SiLU(),nn.Linear(hidden//2,hidden//2)) for _ in range(3)])
        self.other=nn.Sequential(nn.Linear((history_dim-3)*6+context_dim,hidden),nn.SiLU())
        self.score=_ScoreHead(hidden + (hidden//2)*3,hidden)
    def _condition(self,x,c):
        tail=x[:,-6:,:]; chans=[p(tail[:,:,i:i+1]).mean(1) for i,p in enumerate(self.channel_proj)]
        ch=torch.cat(chans,1); other=self.other(torch.cat([tail[:,:,3:].reshape(len(x),-1),c],1)); return torch.cat([ch,other],1)
    def fit(self,x,c,target,persistence,epochs=6,batch_size=512,lr=8e-4):
        seed_everything(self.seed); xt,ct=torch.from_numpy(x).float(),torch.from_numpy(c).float(); r=target-persistence; self._scale=float(torch.from_numpy(r).std().clamp_min(.05)); ds=DataLoader(TensorDataset(xt,ct,torch.from_numpy(r).float()),batch_size=batch_size,shuffle=True)
        opt=torch.optim.AdamW(self.parameters(),lr=lr); hist=[]
        for ep in range(1,epochs+1):
            ls=[]
            for xb,cb,rb in ds:
                t=torch.randint(0,self.schedule.steps,(len(xb),)); ab=self.schedule.alpha_bar[t]; n=torch.randn_like(rb); z=ab.sqrt()*(rb/self._scale)+(1-ab).sqrt()*n; cond=self._condition(xb,cb); pred=self.score(z,t.float()/max(self.schedule.steps-1,1),cond)
                loss=(pred-n).pow(2).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(),1.); opt.step(); ls.append(float(loss.detach()))
            hist.append({"epoch":ep,"loss":float(np.mean(ls)),"residual_scale":self._scale})
        return hist
    def parameters(self): return list(self.channel_proj.parameters())+list(self.other.parameters())+list(self.score.parameters())
    @torch.no_grad()
    def sample(self,x,c,persistence,n_samples=16):
        cond=self._condition(torch.from_numpy(x).float(),torch.from_numpy(c).float()).numpy(); return self._diffusion_sample(self.score,cond,persistence,self._scale,self.schedule,n_samples,self.seed)


class TSDEBranch(_BaseBranch):
    """TSDE core: mask-conditioned recurrent representation with IIF masks."""
    name="TSDE"
    def __init__(self,history_dim,context_dim,hidden=64,steps=8,seed=42):
        self.seed=seed; self.schedule=DiffusionSchedule(steps); self.gru=nn.GRU(history_dim*2,hidden,batch_first=True,bidirectional=True); self.ctx=nn.Linear(context_dim,hidden*2); self.score=_ScoreHead(hidden*4,hidden)
    def _condition(self,x,c,mask):
        h,_=self.gru(torch.cat([x,mask],2)); return torch.cat([h[:,-1],self.ctx(c)],1)
    def fit(self,x,c,target,persistence,epochs=6,batch_size=512,lr=8e-4):
        seed_everything(self.seed); xt,ct=torch.from_numpy(x).float(),torch.from_numpy(c).float(); r=target-persistence; self._scale=float(torch.from_numpy(r).std().clamp_min(.05)); ds=DataLoader(TensorDataset(xt,ct,torch.from_numpy(r).float()),batch_size=batch_size,shuffle=True); opt=torch.optim.AdamW(self.parameters(),lr=lr); hist=[]
        for ep in range(1,epochs+1):
            ls=[]
            for xb,cb,rb in ds:
                # IIF: random point, contiguous interpolation, or tail forecast mask.
                choice=int(torch.randint(0,3,(1,))); mask=torch.ones_like(xb)
                if choice==0: mask=(torch.rand_like(xb)>.2).float()
                elif choice==1: mask[:,8:16,:]=0
                else: mask[:,-6:,:]=0
                t=torch.randint(0,self.schedule.steps,(len(xb),)); ab=self.schedule.alpha_bar[t]; n=torch.randn_like(rb); z=ab.sqrt()*(rb/self._scale)+(1-ab).sqrt()*n; cond=self._condition(xb*mask,cb,mask); pred=self.score(z,t.float()/max(self.schedule.steps-1,1),cond)
                loss=(pred-n).pow(2).mean()+0.02*(cond[:,:-1].mean(1)-cond[:,:-1].mean(1).detach()).pow(2).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(),1.); opt.step(); ls.append(float(loss.detach()))
            hist.append({"epoch":ep,"loss":float(np.mean(ls)),"residual_scale":self._scale,"mask_protocol":"IIF point/interpolation/forecast"})
        return hist
    def parameters(self): return list(self.gru.parameters())+list(self.ctx.parameters())+list(self.score.parameters())
    @torch.no_grad()
    def sample(self,x,c,persistence,n_samples=16):
        xt=torch.from_numpy(x).float(); mask=torch.ones_like(xt); cond=self._condition(xt,torch.from_numpy(c).float(),mask).numpy(); return self._diffusion_sample(self.score,cond,persistence,self._scale,self.schedule,n_samples,self.seed)


class NsDiffBranch(_BaseBranch):
    """NsDiff core: local non-stationary normalization with state FiLM."""
    name="NsDiff"
    def __init__(self,history_dim,context_dim,hidden=64,steps=8,seed=42):
        self.seed=seed; self.schedule=DiffusionSchedule(steps); self.enc=nn.Sequential(nn.Linear(history_dim*6+context_dim+2,hidden),nn.SiLU(),nn.Linear(hidden,hidden)); self.score=_ScoreHead(hidden,hidden)
    def _parts(self,x,c):
        tail=x[:,-6:,:]; m=tail[:,:,0]; mu=m.mean(1); sig=m.std(1).clamp_min(.05); norm=(m-mu[:,None])/sig[:,None]; feat=torch.cat([tail.reshape(len(x),-1),c,mu[:,None],sig[:,None]],1); return feat,mu,sig,norm
    def fit(self,x,c,target,persistence,epochs=6,batch_size=512,lr=8e-4):
        seed_everything(self.seed); xt,ct=torch.from_numpy(x).float(),torch.from_numpy(c).float(); yt=torch.from_numpy(target).float(); feat,mu,sig,_=self._parts(xt,ct); base=mu; r=(yt-base); self._scale=float(r.std().clamp_min(.05)); ds=DataLoader(TensorDataset(feat,r),batch_size=batch_size,shuffle=True); opt=torch.optim.AdamW(self.parameters(),lr=lr); hist=[]
        for ep in range(1,epochs+1):
            ls=[]
            for fb,rb in ds:
                t=torch.randint(0,self.schedule.steps,(len(fb),)); ab=self.schedule.alpha_bar[t]; n=torch.randn_like(rb); z=ab.sqrt()*(rb/self._scale)+(1-ab).sqrt()*n; pred=self.score(z,t.float()/max(self.schedule.steps-1,1),self.enc(fb)); loss=(pred-n).pow(2).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.parameters(),1.); opt.step(); ls.append(float(loss.detach()))
            hist.append({"epoch":ep,"loss":float(np.mean(ls)),"residual_scale":self._scale})
        return hist
    def parameters(self): return list(self.enc.parameters())+list(self.score.parameters())
    @torch.no_grad()
    def sample(self,x,c,persistence,n_samples=16):
        xt=torch.from_numpy(x).float(); ct=torch.from_numpy(c).float(); feat,mu,_,_=self._parts(xt,ct); return self._diffusion_sample(self.score,self.enc(feat).numpy(),mu.numpy(),self._scale,self.schedule,n_samples,self.seed)


BRANCHES={"TimeGrad":TimeGradBranch,"DiffPLF":DiffPLFBranch,"CCDM":CCDMBranch,"TSDE":TSDEBranch,"NsDiff":NsDiffBranch}
