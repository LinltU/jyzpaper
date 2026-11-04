"""Stable-normal baseline + conditional residual diffusion for experiment data.

Deployment target: binary alert only (normal vs. any zero-value insulator).
Only error==0 windows are used for fitting. error!=0 is evaluation metadata.
The diffusion model generates a small number of normal residual vectors, not
raw Measure values and not error-position samples.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_curve

F=["M4","M7","Temp_shared","Hum_shared","Mdiff","Mratio","logratio","elapsed_min"]

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

def load(path,L):
    ss={p.stem:pd.read_csv(p) for p in Path(path).glob("*.csv")}; groups={}
    for n,d in ss.items(): groups.setdefault(n[:6],{})[int(n.split("(")[1][0])]=d
    xs=[]; meta=[]
    for date,p in sorted(groups.items()):
        a,b=p[4].copy(),p[7].copy(); a["dt"]=pd.to_datetime(a.time,format="%Y-%m-%d_%H:%M:%S"); b["dt"]=pd.to_datetime(b.time,format="%Y-%m-%d_%H:%M:%S"); a=a.sort_values("dt").reset_index(drop=True); b=b.sort_values("dt").reset_index(drop=True)
        bm=pd.merge_asof(a[["dt"]],b,on="dt",direction="nearest",tolerance=pd.Timedelta("40s")); keep=bm.Measure.notna().values; a=a.loc[keep].reset_index(drop=True); bm=bm.loc[keep].reset_index(drop=True); t0=a.dt.iloc[0]; elapsed=(a.dt-t0).dt.total_seconds()/60
        z=pd.DataFrame({"M4":a.Measure.values,"M7":bm.Measure.values,"Temp_shared":(a.Temp.values+bm.Temp.values)/2,"Hum_shared":(a.Hum.values+bm.Hum.values)/2,"Mdiff":a.Measure.values-bm.Measure.values,"Mratio":(a.Measure.values+1e-3)/(bm.Measure.values+1e-3),"logratio":np.log((a.Measure.values+1e-3)/(bm.Measure.values+1e-3)),"elapsed_min":elapsed.values,"error":a.error.values})
        st=0
        for i in range(1,len(z)+1):
            if i==len(z) or int(z.error.iloc[i])!=int(z.error.iloc[st]):
                run=z.iloc[st:i]
                for j in range(max(0,len(run)-L+1)):
                    q=run.iloc[j:j+L]; xs.append(q[F].to_numpy(np.float32)); meta.append({"date":date,"error":int(q.error.iloc[0]),"start":int(st+j)})
                st=i
    return np.stack(xs),pd.DataFrame(meta)

def stable_mask(x, meta):
    # Stable means low late-window slope and low local variation in both
    # measured channels. Thresholds are learned per date from normal windows.
    scores=np.zeros(len(x)); slope4=[]; slope7=[]; sd4=[]; sd7=[]; t=np.arange(x.shape[1])
    for q in x:
        slope4.append(abs(np.polyfit(t,q[:,0],1)[0])); slope7.append(abs(np.polyfit(t,q[:,1],1)[0])); sd4.append(q[:,0].std()); sd7.append(q[:,1].std())
    slope4=np.asarray(slope4); slope7=np.asarray(slope7); sd4=np.asarray(sd4); sd7=np.asarray(sd7)
    for date in sorted(meta.date.unique()):
        ids=np.where((meta.date.values==date)&(meta.error.values==0))[0]
        if len(ids)==0: continue
        # Use lower 40% of a combined dynamic score as stable candidates.
        sc=(slope4[ids]/np.quantile(slope4[ids],.75).clip(1e-5)+slope7[ids]/np.quantile(slope7[ids],.75).clip(1e-5)+sd4[ids]/np.quantile(sd4[ids],.75).clip(1e-5)+sd7[ids]/np.quantile(sd7[ids],.75).clip(1e-5))
        cutoff=np.quantile(sc,.40); scores[ids]=sc; scores[ids[sc<=cutoff]]=0
    return (scores==0)&(meta.error.values==0), {"stable_windows":int(((scores==0)&(meta.error.values==0)).sum()),"all_normal_windows":int((meta.error.values==0).sum()),"stable_definition":"per-date lowest 40% combined absolute slope and std of M4/M7"}

def design(x):
    q=x[:,-1,:]; e=q[:,7]; return np.c_[np.ones(len(x)),q[:,2],q[:,3],e,q[:,2]**2,q[:,3]**2,q[:,2]*q[:,3]]

def residuals(train, test):
    X=design(train); Z=design(test); y4=train[:,-1,0]; y7=train[:,-1,1]; yd=train[:,-1,4]; yl=train[:,-1,6]
    bs=[np.linalg.lstsq(X,y,rcond=None)[0] for y in [y4,y7,yd,yl]]; return np.c_[y4-X@bs[0],y7-X@bs[1],yd-X@bs[2],yl-X@bs[3]],np.c_[test[:,-1,0]-Z@bs[0],test[:,-1,1]-Z@bs[1],test[:,-1,4]-Z@bs[2],test[:,-1,6]-Z@bs[3]],bs

def mahal(train_r,test_r,extra=None):
    r=train_r if extra is None else np.vstack([train_r,extra]); mu=r.mean(0); cov=np.cov(r,rowvar=False)+np.eye(r.shape[1])*1e-5; inv=np.linalg.pinv(cov); d=test_r-mu; return np.einsum('ij,jk,ik->i',d,inv,d)

class DDPM(nn.Module):
    def __init__(self,d,c,steps=16):
        super().__init__(); self.steps=steps; self.net=nn.Sequential(nn.Linear(d+c+32,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,d)); beta=torch.linspace(1e-4,.02,steps); self.register_buffer('beta',beta); self.register_buffer('alpha',1-beta); self.register_buffer('abar',torch.cumprod(1-beta,0))
    def emb(self,t):
        f=torch.exp(-math.log(10000)*torch.arange(16,device=t.device)/15); z=t[:,None].float()*f[None,:]; return torch.cat([z.sin(),z.cos()],1)
    def forward(self,z,t,c): return self.net(torch.cat([z,c,self.emb(t)],1))

def train_diff(r,cond,epochs,seed):
    seed_all(seed); mu=r.mean(0); sd=r.std(0).clip(.01); y=torch.tensor((r-mu)/sd,dtype=torch.float32); c=torch.tensor(cond,dtype=torch.float32); m=DDPM(r.shape[1],cond.shape[1]); opt=torch.optim.AdamW(m.parameters(),lr=8e-4); dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y,c),batch_size=min(64,len(y)),shuffle=True)
    for _ in range(epochs):
        for b,cb in dl:
            t=torch.randint(0,m.steps,(len(b),)); ab=m.abar[t]; n=torch.randn_like(b); z=ab[:,None].sqrt()*b+(1-ab[:,None]).sqrt()*n; loss=(m(z,t,cb)-n).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return m,mu,sd

@torch.no_grad()
def sample(m,mu,sd,cond,seed):
    c=torch.tensor(cond,dtype=torch.float32); g=torch.Generator().manual_seed(seed); z=torch.randn((len(c),m.net[-1].out_features),generator=g)
    for k in range(m.steps-1,-1,-1):
        t=torch.full((len(c),),k,dtype=torch.long); e=m(z,t,c); a,ab=m.alpha[k],m.abar[k]; z=(z-(1-a)/(1-ab).sqrt()*e)/a.sqrt();
        if k>0: z=z+m.beta[k].sqrt()*torch.randn(z.shape,generator=g)
    return z.numpy()*sd+mu

def score_eval(s,y):
    auc=roc_auc_score(y,s); ap=average_precision_score(y,s); p,r,t=precision_recall_curve(y,s); f=2*p*r/(p+r+1e-9); ix=int(np.argmax(f)); th=float(t[min(ix,len(t)-1)] if len(t) else np.inf); pred=(s>=th).astype(int); return {"auroc":float(auc),"auprc":float(ap),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"normal_false_alarm":float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum())),"threshold":th}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path(r'C:\Project\JYZmentor\dataset\experiment_sheets_csv')); ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs'/'stable_residual_diffusion_20260909'); ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    x,meta=load(args.input,6); stable,stable_info=stable_mask(x,meta); dates=sorted(meta.date.unique()); results={'protocol':{'target':'error==0 vs error!=0','features':F,'stable_info':stable_info,'train_rule':'only stable error==0 windows','augmentation_rule':'generate only normal residual vectors, 20% of stable normal count'}}; folds=[]
    for td in dates:
        tr=np.where(meta.date.values!=td)[0]; te=np.where(meta.date.values==td)[0]; ntr=tr[stable[tr]]; y=(meta.error.values[te]!=0).astype(int); rtr,rte,_=residuals(x[ntr],x[te]); fold={'test_date':td,'stable_train':int(len(ntr)),'test_windows':int(len(te)),'positive_test':int(y.sum())}
        fold['Stable_Mahalanobis']=score_eval(mahal(rtr,rte),y)
        # condition on shared temperature, humidity and elapsed of stable windows
        ctrain=x[ntr][:,-1,[2,3,7]]; ctest=x[te][:,-1,[2,3,7]]; cmean=ctrain.mean(0); cstd=ctrain.std(0).clip(.1); ctrain=(ctrain-cmean)/cstd; ctest=(ctest-cmean)/cstd
        m,rm,rs=train_diff(rtr,ctrain,args.epochs,args.seed); ns=max(1,int(len(rtr)*.2)); syn=sample(m,rm,rs,ctrain[:ns],args.seed+1); fold['Stable_Residual_DDPM_augmented']=score_eval(mahal(rtr,rte,syn),y); fold['synthetic_normal_residuals']=int(ns); folds.append(fold)
    results['folds']=folds; methods=['Stable_Mahalanobis','Stable_Residual_DDPM_augmented']; results['summary']={m:{k:float(np.nanmean([f[m][k] for f in folds])) for k in ['auroc','auprc','balanced_accuracy','normal_false_alarm']} for m in methods}; (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8'); lines=['# 稳定正常段 + 双设备残差扩散', '', '| 方法 | AUROC | AUPRC | 平衡准确率 | 正常误报率 |','|---|---:|---:|---:|---:|'];
    for m,v in results['summary'].items(): lines.append(f"| {m} | {v['auroc']:.4f} | {v['auprc']:.4f} | {v['balanced_accuracy']:.4f} | {v['normal_false_alarm']:.4f} |")
    lines += ['', '只使用后期稳定的 error=0 窗口；扩散模型只生成正常双设备残差，不生成异常片号或原始 Measure。结果按完整实验日期留出测试。']; (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8'); print(json.dumps(results['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__': main()
