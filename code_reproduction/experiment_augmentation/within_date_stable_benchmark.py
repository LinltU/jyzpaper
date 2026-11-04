"""Within-experiment-date stable baseline benchmark.

This is the next step after the cross-date experiment: calibrate the normal
distribution separately for each date, because absolute stable Measure levels
drift between laboratory runs. The detector is binary (error==0 vs error!=0).
Only stable normal windows before/within the normal run are fitted; error is
used only for offline scoring.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score
from stable_residual_diffusion import load, stable_mask, residuals, mahal, train_diff, sample
from normal_dual_device_benchmark import detrend_score, train_ae, ae_score

def ev(score,y):
    auc=roc_auc_score(y,score) if len(np.unique(y))>1 else float('nan'); ap=average_precision_score(y,score) if y.sum()>0 else float('nan'); p,r,t=precision_recall_curve(y,score); f=2*p*r/(p+r+1e-9); i=int(np.nanargmax(f)); th=float(t[min(i,len(t)-1)]) if len(t) else float('inf'); pred=(score>=th).astype(int); return {'auroc':float(auc),'average_precision':float(ap),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'normal_false_alarm':float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum())),'threshold':th}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path(r'C:\Project\JYZmentor\dataset\experiment_sheets_csv')); ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs'/'within_date_stable_20260909'); ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    x,meta=load(args.input,6); stable,info=stable_mask(x,meta); folds=[]
    for date in sorted(meta.date.unique()):
        ids=np.where(meta.date.values==date)[0]; normals=ids[(meta.error.values[ids]==0)&stable[ids]]; test=ids
        # Restrict calibration to the earliest stable normal windows. This
        # prevents later abnormal segments from influencing the baseline.
        normals=normals[np.argsort(meta.start.values[normals])]; nfit=max(8,int(len(normals)*.7)); fit=normals[:nfit]
        rtr,rte,_=residuals(x[fit],x[test]); y=(meta.error.values[test]!=0).astype(int); fold={'date':date,'fit_stable_normal':int(len(fit)),'test_windows':int(len(test)),'positive_test':int(y.sum())}
        fold['within_date_stable_mahalanobis']=ev(mahal(rtr,rte),y)
        fold['within_date_D3R_style']=ev(detrend_score(x[fit],x[test]),y)
        # Local DiffAD/IMDiffusion/ProDiffAD/TIMEADDM mechanisms share the
        # normal-only reconstruction protocol but use different masking or
        # score weighting in the main benchmark. Here we run the two most
        # directly comparable stable-window reconstruction variants.
        # The helper expects scaled data statistics; use local fit statistics.
        flat=x[fit].reshape(len(fit),-1); am=flat.mean(0); ass=flat.std(0).clip(.03); ae=train_ae(x[fit],am,ass,masked=False,epochs=args.epochs,seed=args.seed)
        fold['within_date_DiffAD_style']=ev(ae_score(ae,x[test],am,ass),y)
        mae=train_ae(x[fit],am,ass,masked=True,epochs=args.epochs,seed=args.seed)
        fold['within_date_IMDiffusion_style']=ev(ae_score(mae,x[test],am,ass),y)
        # Normal-residual DDPM augmentation, limited to 20% of the fit set.
        cond=x[fit][:,-1,[2,3,7]]; cm=cond.mean(0); cs=cond.std(0).clip(.1); c=(cond-cm)/cs; m,rm,rs=train_diff(rtr,c,args.epochs,args.seed); nsynt=max(1,int(len(rtr)*.2)); syn=sample(m,rm,rs,c[:nsynt],args.seed+1); fold['within_date_stable_mahalanobis_augmented']=ev(mahal(rtr,rte,syn),y); fold['synthetic_normal_residuals']=nsynt; folds.append(fold)
    methods=['within_date_stable_mahalanobis','within_date_D3R_style','within_date_DiffAD_style','within_date_IMDiffusion_style','within_date_stable_mahalanobis_augmented']; summary={m:{k:float(np.nanmean([f[m][k] for f in folds])) for k in ['auroc','average_precision','balanced_accuracy','precision','recall','f1','normal_false_alarm']} for m in methods}; payload={'protocol':{'target':'error==0 vs error!=0','calibration':'per experiment date, earliest stable normal windows','augmentation':'20% normal residuals only','stable_info':info},'folds':folds,'summary':summary}; (args.output/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); lines=['# 按实验日期校准的稳定正常基线','','| 方法 | AUROC | AUPRC | 平衡准确率 | 精确率 | 召回率 | F1 | 正常误报率 |','|---|---:|---:|---:|---:|---:|---:|---:|'];
    for m,v in summary.items(): lines.append(f"| {m} | {v['auroc']:.4f} | {v['average_precision']:.4f} | {v['balanced_accuracy']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['normal_false_alarm']:.4f} |")
    lines += ['', '每个日期单独使用最早的稳定正常窗口校准，error!=0 只用于评估。扩散模型仅生成正常残差，未生成异常数据。']; (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

