"""1-D dual-device relational conditional diffusion for experiment.xlsx.

The denoised target is a four-channel relational residual sequence:
M4 residual, M7 residual, M4-M7 residual and log(M4/M7) residual.
Only stable normal windows from each date calibrate the baseline.  The model
is conditioned on shared temperature/humidity and their short-term trends.
This is a local adaptation of TimeGrad/NsDiff-style conditional diffusion,
not a claim of exact reproduction of the upstream repositories.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

ROOT=Path(__file__).resolve().parent
from run_experiment_augmentation import load_windows, seed_all

class RelDDPM(nn.Module):
    def __init__(self, dim, cond_dim, hidden=192, steps=20):
        super().__init__(); self.steps=steps
        self.net=nn.Sequential(nn.Linear(dim+cond_dim+32,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,dim))
        beta=torch.linspace(1e-4,.02,steps); self.register_buffer('beta',beta); self.register_buffer('alpha',1-beta); self.register_buffer('abar',torch.cumprod(1-beta,0))
    def emb(self,t):
        h=16; f=torch.exp(-math.log(10000)*torch.arange(h,device=t.device)/(h-1)); z=t[:,None].float()*f[None,:]; return torch.cat([z.sin(),z.cos()],1)
    def forward(self,z,t,c): return self.net(torch.cat([z,c,self.emb(t)],1))

def make_features(x, features):
    # shared environment and relational channels are present in the source windows
    idx={k:i for i,k in enumerate(features)}
    return x[:,:, [idx['M4'],idx['M7'],idx['Mdiff'],idx['logratio']]]

def make_cond(x, features):
    idx={k:i for i,k in enumerate(features)}; temp=x[:,:,idx['Temp_shared']]; hum=x[:,:,idx['Hum_shared']]
    c=np.stack([temp[:,-1],hum[:,-1],temp[:,-1]-temp[:,0],hum[:,-1]-hum[:,0]],1)
    return c.astype(np.float32)

def fit(y,c,epochs,seed):
    seed_all(seed); n=y.shape[0]; flat=y.reshape(n,-1); mu=flat.mean(0); sd=flat.std(0).clip(.03); z=torch.tensor((flat-mu)/sd,dtype=torch.float32); ct=torch.tensor(c,dtype=torch.float32)
    model=RelDDPM(z.shape[1],ct.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=8e-4); ds=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(z,ct),batch_size=min(96,n),shuffle=True)
    hist=[]; model.train()
    for _ in range(epochs):
        ls=[]
        for xb,cb in ds:
            t=torch.randint(0,model.steps,(len(xb),)); ab=model.abar[t]; noise=torch.randn_like(xb); noisy=ab[:,None].sqrt()*xb+(1-ab[:,None]).sqrt()*noise; pred=model(noisy,t,cb); loss=(pred-noise).pow(2).mean(); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); ls.append(float(loss))
        hist.append(float(np.mean(ls)))
    return model,mu,sd,hist

@torch.no_grad()
def score(model, y, c, mu, sd):
    model.eval(); z=torch.tensor(((y.reshape(len(y),-1)-mu)/sd),dtype=torch.float32); ct=torch.tensor(c,dtype=torch.float32); vals=[]
    for t in range(model.steps):
        tt=torch.full((len(z),),t,dtype=torch.long); ab=model.abar[tt]; noise=torch.randn_like(z); noisy=ab[:,None].sqrt()*z+(1-ab[:,None]).sqrt()*noise; pred=model(noisy,tt,ct); vals.append(((pred-noise)**2).mean(1).numpy())
    return np.mean(vals,axis=0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--epochs',type=int,default=120); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    features=['M4','M7','Temp_shared','Hum_shared','Mdiff','Mratio','logratio','elapsed_min']; x,meta=load_windows(args.input,6,features); rel=make_features(x,features); cond=make_cond(x,features); stable=np.zeros(len(meta),bool)
    # per-date stable normal calibration: lowest 40% combined slope/std
    for d in sorted(meta.date.unique()):
        ids=np.where((meta.date.values==d)&(meta.error.values==0))[0]; dyn=rel[ids].std(1).mean(1)+np.abs(rel[ids][:,-1]-rel[ids][:,0]).mean(1); stable[ids[dyn<=np.quantile(dyn,.4)]]=True
    train=[]; test=[]
    for d in sorted(meta.date.unique()):
        ids=np.where(meta.date.values==d)[0]; nids=ids[(meta.error.values[ids]==0)&stable[ids]]; nids=nids[np.argsort(meta.start_index.values[nids])]; cut=max(8,int(len(nids)*.7)); train.extend(nids[:cut]); test.extend(ids)
    train=np.array(sorted(set(train))); test=np.array(test); base=rel[train].mean((0,1)); ytr=rel[train]-base; yte=rel[test]-base; ctrain=cond[train]; ctest=cond[test]; model,mu,sd,hist=fit(ytr,ctrain,args.epochs,args.seed); sc=score(model,yte,ctest,mu,sd); y=(meta.error.values[test]!=0).astype(int); threshold=np.quantile(sc[y==0],.995); pred=(sc>threshold).astype(int)
    out={'protocol':{'source':str(args.input),'target':'dual-device relational residual sequence','features':['M4_residual','M7_residual','Mdiff_residual','logratio_residual'],'conditions':['Temp_shared','Hum_shared','TempTrend','HumidityTrend'],'train_stable_normals':int(len(train)),'test_real_windows':int(len(test)),'epochs':args.epochs,'paper_adaptation':['TimeGrad','NsDiff','DiffAD normal-only boundary']},'metrics':{'auroc':float(roc_auc_score(y,sc)),'auprc':float(average_precision_score(y,sc)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'normal_false_alarm':float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum())),'threshold':float(threshold)},'training':{'final_loss':hist[-1]}}
    (args.output/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); pd.DataFrame({'score':sc,'label':y,'date':meta.date.values[test],'error':meta.error.values[test]}).to_csv(args.output/'window_scores.csv',index=False); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
