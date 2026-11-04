"""Normal-only dual-device anomaly benchmark for experiment.xlsx.

The deployment task is binary: error==0 is normal and error!=0 means a
zero-value insulator exists somewhere in the string. Models are trained only
on normal windows. The error column is used only for offline evaluation.

The paper-inspired labels are deliberately explicit:
  - Mahalanobis: normal-distribution baseline.
  - D3R-style: detrend-before-scoring reconstruction residual.
  - DiffAD/IMDiffusion-style: masked denoising autoencoder and ensemble score.
  - NsDiff-style augmentation: local normalization + DDPM normal-window
    generation; synthetic normal windows are added to the denoising model.
These are local mechanisms adapted to this workbook, not official benchmark
repository runs.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_curve, precision_score, recall_score

FEATURES=["M4","M7","Temp_shared","Hum_shared","Mdiff","Mratio","logratio","elapsed_min"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

def load_windows(path,length=6):
    sheets={p.stem:pd.read_csv(p) for p in Path(path).glob("*.csv")}
    groups={}
    for name,d in sheets.items(): groups.setdefault(name[:6],{})[int(name.split("(")[1][0])]=d
    xs=[]; ms=[]
    for date,pair in sorted(groups.items()):
        a,b=pair[4].copy(),pair[7].copy(); a["dt"]=pd.to_datetime(a.time,format="%Y-%m-%d_%H:%M:%S"); b["dt"]=pd.to_datetime(b.time,format="%Y-%m-%d_%H:%M:%S")
        a=a.sort_values("dt").reset_index(drop=True); b=b.sort_values("dt").reset_index(drop=True)
        bm=pd.merge_asof(a[["dt"]],b,on="dt",direction="nearest",tolerance=pd.Timedelta("40s")); keep=bm.Measure.notna().values; a=a.loc[keep].reset_index(drop=True); bm=bm.loc[keep].reset_index(drop=True)
        t0=a.dt.iloc[0]; elapsed=(a.dt-t0).dt.total_seconds()/60
        z=pd.DataFrame({"M4":a.Measure.values,"M7":bm.Measure.values,"Temp_shared":(a.Temp.values+bm.Temp.values)/2,"Hum_shared":(a.Hum.values+bm.Hum.values)/2,"Mdiff":a.Measure.values-bm.Measure.values,"Mratio":(a.Measure.values+1e-3)/(bm.Measure.values+1e-3),"logratio":np.log((a.Measure.values+1e-3)/(bm.Measure.values+1e-3)),"elapsed_min":elapsed.values,"error":a.error.values})
        st=0
        for i in range(1,len(z)+1):
            if i==len(z) or int(z.error.iloc[i])!=int(z.error.iloc[st]):
                run=z.iloc[st:i]
                for j in range(max(0,len(run)-length+1)):
                    q=run.iloc[j:j+length]; xs.append(q[FEATURES].to_numpy(np.float32)); ms.append({"date":date,"error":int(q.error.iloc[0]),"start":int(st+j)})
                st=i
    return np.stack(xs),pd.DataFrame(ms)

class AE(nn.Module):
    def __init__(self,dim,hidden=128):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,dim))
    def forward(self,x): return self.net(x)

class CondDDPM(nn.Module):
    def __init__(self,dim,cond_dim,steps=16,hidden=192):
        super().__init__(); self.steps=steps; self.net=nn.Sequential(nn.Linear(dim+cond_dim+32,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,dim)); beta=torch.linspace(1e-4,.02,steps); self.register_buffer("beta",beta); self.register_buffer("alpha",1-beta); self.register_buffer("abar",torch.cumprod(1-beta,0))
    def emb(self,t):
        f=torch.exp(-math.log(10000)*torch.arange(16,device=t.device)/15); z=t[:,None].float()*f[None,:]; return torch.cat([z.sin(),z.cos()],1)
    def forward(self,z,t,c): return self.net(torch.cat([z,c,self.emb(t)],1))

def norm_fit(x):
    flat=x.reshape(len(x),-1); mu=flat.mean(0); sd=flat.std(0).clip(.03); return mu,sd

def train_ae(x,mu,sd,masked=False,extra=None,epochs=60,seed=42):
    seed_all(seed); flat=x.reshape(len(x),-1); y=torch.tensor((flat-mu)/sd,dtype=torch.float32); 
    if extra is not None: y=torch.cat([y,torch.tensor((extra.reshape(len(extra),-1)-mu)/sd,dtype=torch.float32)],0)
    model=AE(y.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); dl=torch.utils.data.DataLoader(y,batch_size=min(128,len(y)),shuffle=True)
    for _ in range(epochs):
        for b in dl:
            inp=b
            if masked:
                mask=(torch.rand_like(b)>.15).float(); inp=b*mask
            pred=model(inp); loss=(pred-b).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def ae_score(model,x,mu,sd):
    y=torch.tensor((x.reshape(len(x),-1)-mu)/sd,dtype=torch.float32); return ((model(y)-y)**2).mean(1).numpy()

def detrend_score(train,test):
    # D3R-style dynamic decomposition: remove a per-window linear trend from
    # the two measured channels and score residual dynamics against normal.
    def feat(x):
        n,l,f=x.shape; t=np.arange(l); out=[]
        for q in x:
            r=q.copy();
            for k in [0,1,4,5,6]:
                co=np.polyfit(t,r[:,k],1); r[:,k]-=np.polyval(co,t)
            out.append(np.r_[r.reshape(-1),q[-1,0]-q[0,0],q[-1,1]-q[0,1]])
        return np.asarray(out)
    a,b=feat(train),feat(test); mu=a.mean(0); sd=a.std(0).clip(.03); return (((b-mu)/sd)**2).mean(1)

def conditional_residual_score(train,test):
    """Normal baseline: predict the two measures from shared environment and
    elapsed time, then score the paired residuals and relation features."""
    def design(x):
        # flatten the window but keep local environmental trajectory summary;
        # the last point is the decision point used by the alert.
        env=x[:,-1,:][:,[2,3,7]]
        return np.c_[np.ones(len(x)),env,env[:,0]**2,env[:,1]**2,env[:,0]*env[:,1]]
    X=design(train); Z=design(test); y4=train[:,-1,0]; y7=train[:,-1,1]
    b4=np.linalg.lstsq(X,y4,rcond=None)[0]; b7=np.linalg.lstsq(X,y7,rcond=None)[0]
    r4=y4-X@b4; r7=y7-X@b7; s4=np.std(r4).clip(.01); s7=np.std(r7).clip(.01)
    q4=test[:,-1,0]-Z@b4; q7=test[:,-1,1]-Z@b7
    # Include the pair relation so a common-mode environment shift does not
    # dominate the score.
    m=train[:,-1,4]; mt=test[:,-1,4]; bm=np.linalg.lstsq(X,m,rcond=None)[0]; rm=m-X@bm; sm=np.std(rm).clip(.01); qm=mt-Z@bm
    return (q4/s4)**2*.4+(q7/s7)**2*.4+(qm/sm)**2*.2

def train_ddpm_normal(x,mu,sd,dates,meta,epochs=60,seed=42):
    seed_all(seed); flat=(x.reshape(len(x),-1)-mu)/sd; c=np.array([[1. if r.date==d else 0. for d in dates] for _,r in meta.iterrows()],np.float32); y=torch.tensor(flat,dtype=torch.float32); ct=torch.tensor(c,dtype=torch.float32); m=CondDDPM(y.shape[1],ct.shape[1]); opt=torch.optim.AdamW(m.parameters(),lr=8e-4); dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y,ct),batch_size=min(128,len(y)),shuffle=True)
    for _ in range(epochs):
        for b,cb in dl:
            t=torch.randint(0,m.steps,(len(b),)); ab=m.abar[t]; n=torch.randn_like(b); z=ab[:,None].sqrt()*b+(1-ab[:,None]).sqrt()*n; loss=(m(z,t,cb)-n).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def sample_ddpm(m,n,dates,mu,sd,seed=42):
    c=torch.tensor([[1. if d==dates[i%len(dates)] else 0. for d in dates] for i in range(n)],dtype=torch.float32); out=[]
    for j in range(2):
        g=torch.Generator().manual_seed(seed+j); z=torch.randn((n,m.net[-1].out_features),generator=g)
        for k in range(m.steps-1,-1,-1):
            t=torch.full((n,),k,dtype=torch.long); e=m(z,t,c); a,ab=m.alpha[k],m.abar[k]; z=(z-(1-a)/(1-ab).sqrt()*e)/a.sqrt();
            if k>0: z=z+m.beta[k].sqrt()*torch.randn(z.shape,generator=g)
        out.append(z.numpy()*sd+mu)
    return out[0].reshape(n,6,len(FEATURES))

def evaluate(scores,y):
    auc=float(roc_auc_score(y,scores)) if len(np.unique(y))>1 else float("nan"); ap=float(average_precision_score(y,scores)) if y.sum()>0 else float("nan");
    p,r,t=precision_recall_curve(y,scores); f=np.nan_to_num(2*p*r/(p+r+1e-9)); ix=int(np.argmax(f)); thr=float(t[min(ix,len(t)-1)]) if len(t) else float("inf"); pred=(scores>=thr).astype(int); ba=float(balanced_accuracy_score(y,pred)); return {"auroc":auc,"auprc":ap,"best_f1_threshold":thr,"balanced_accuracy_at_best_f1":ba,"precision_at_best_f1":float(precision_score(y,pred,zero_division=0)),"recall_at_best_f1":float(recall_score(y,pred,zero_division=0)),"normal_false_alarm_at_threshold":float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum()))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=Path(r"C:\Project\JYZmentor\dataset\experiment_sheets_csv")); ap.add_argument("--output",type=Path,default=Path(__file__).resolve().parent/"runs"/"normal_dual_device_20260909"); ap.add_argument("--window-length",type=int,default=6); ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    x,meta=load_windows(args.input,args.window_length); dates=sorted(meta.date.unique()); results={"protocol":{"features":FEATURES,"window_length":args.window_length,"train_only_normal":True,"target":"error==0 vs error!=0","paper_code_boundary":"local mechanism adaptations; not official repository benchmark runs"},"folds":[]}
    for test_date in dates:
        tr=np.where(meta.date.values!=test_date)[0]; te=np.where(meta.date.values==test_date)[0]; normal=tr[meta.error.values[tr]==0]; y=(meta.error.values[te]!=0).astype(int); mu,sd=norm_fit(x[normal]); fold={"test_date":test_date,"train_normal_windows":int(len(normal)),"test_windows":int(len(te)),"positive_test_windows":int(y.sum())}
        fold["Mahalanobis_normal"] = evaluate((((x[te].reshape(len(te),-1)-mu)/sd)**2).mean(1),y)
        fold["Normal_conditional_residual"] = evaluate(conditional_residual_score(x[normal],x[te]),y)
        fold["D3R_style_detrended"] = evaluate(detrend_score(x[normal],x[te]),y)
        ae=train_ae(x[normal],mu,sd,masked=False,epochs=args.epochs,seed=args.seed); fold["DiffAD_style_reconstruction"]=evaluate(ae_score(ae,x[te],mu,sd),y)
        mae=train_ae(x[normal],mu,sd,masked=True,epochs=args.epochs,seed=args.seed); fold["IMDiffusion_style_masked"]=evaluate(ae_score(mae,x[te],mu,sd),y)
        # NsDiff-style normal augmentation: generate only normal windows and
        # add them to the masked denoiser. This never uses abnormal windows.
        nm=meta.iloc[normal]; gen_model=train_ddpm_normal(x[normal],mu,sd,sorted(nm.date.unique()),nm,epochs=args.epochs,seed=args.seed); syn=sample_ddpm(gen_model,max(1,len(normal)//3),sorted(nm.date.unique()),mu,sd,args.seed); mae_aug=train_ae(x[normal],mu,sd,masked=True,extra=syn,epochs=args.epochs,seed=args.seed); fold["NsDiff_style_normal_augmented"]=evaluate(ae_score(mae_aug,x[te],mu,sd),y)
        results["folds"].append(fold)
    methods=[k for k in results["folds"][0] if k not in ["test_date","train_normal_windows","test_windows","positive_test_windows"]]; summary={}
    for m in methods:
        vals=[f[m] for f in results["folds"]]; summary[m]={q:float(np.nanmean([v[q] for v in vals])) for q in ["auroc","auprc","balanced_accuracy_at_best_f1","precision_at_best_f1","recall_at_best_f1","normal_false_alarm_at_threshold"]}
    results["summary"]=summary; (args.output/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 正常双设备模型与零值异常检测", "", "目标：只用 error=0 正常窗口训练，评估 error!=0 是否存在零值绝缘子。", "", "| 方法 | AUROC | AUPRC | 平衡准确率 | 精确率 | 召回率 | 正常误报率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m,v in summary.items(): lines.append(f"| {m} | {v['auroc']:.4f} | {v['auprc']:.4f} | {v['balanced_accuracy_at_best_f1']:.4f} | {v['precision_at_best_f1']:.4f} | {v['recall_at_best_f1']:.4f} | {v['normal_false_alarm_at_threshold']:.4f} |")
    lines += ["", "## 方法边界", "", "Mahalanobis 为正常分布基线；D3R_style_detrended 对应先去动态趋势再评分；DiffAD_style_reconstruction 对应正常窗口重构残差；IMDiffusion_style_masked 对应随机掩码重构集成思想；NsDiff_style_normal_augmented 只生成正常窗口并加入掩码重构训练。均为本地机制适配，不宣称官方代码逐层运行。"]
    (args.output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
