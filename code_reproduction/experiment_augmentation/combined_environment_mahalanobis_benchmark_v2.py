"""Readable fusion of P10/P90 environment score and Mahalanobis score.

The important combination is explicitly implemented in ``fuse_scores``:

    final = environment_weight * environment_z
          + (1 - environment_weight) * mahalanobis_z

Only experiment.xlsx mirror data is used.  ``error`` is used for offline
evaluation only; normal calibration uses stable error==0 windows.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from stable_residual_diffusion import load, stable_mask, residuals, mahal


def environment_conditions(windows):
    """Return [Temp_shared, Hum_shared, TempTrend, HumidityTrend, elapsed]."""
    last, first = windows[:, -1, :], windows[:, 0, :]
    d=max(1, windows.shape[1]-1)
    return np.c_[last[:,2], last[:,3], (last[:,2]-first[:,2])/d, (last[:,3]-first[:,3])/d, last[:,7]]


def interval_exceedance(values, bands):
    """Score how far M4/M7 residuals are outside P10-P90 intervals."""
    scores=[]
    for channel in range(2):
        p10, p90 = bands[:,channel,0], bands[:,channel,2]
        width=np.maximum(p90-p10,.03)
        scores.append(np.maximum((p10-values[:,channel])/width,0)+np.maximum((values[:,channel]-p90)/width,0))
    return np.maximum(scores[0],scores[1])


def fit_quantile_score(xfit,rfit,xtest,rtest,edges):
    """Fit P10/P50/P90 regressors separately by humidity stage."""
    cf,ct=environment_conditions(xfit),environment_conditions(xtest)
    sf,st=np.digitize(cf[:,1],edges),np.digitize(ct[:,1],edges)
    tb=np.zeros((len(xtest),2,3)); fb=np.zeros((len(xfit),2,3))
    for ch in range(2):
        for qi,alpha in enumerate((.1,.5,.9)):
            global_model=GradientBoostingRegressor(loss='quantile',alpha=alpha,n_estimators=80,max_depth=2,learning_rate=.04,random_state=42).fit(cf,rfit[:,ch])
            for stage in range(3):
                ids=np.where(sf==stage)[0]
                if len(ids)>=12:
                    model=GradientBoostingRegressor(loss='quantile',alpha=alpha,n_estimators=60,max_depth=2,learning_rate=.04,random_state=42).fit(cf[ids],rfit[ids,ch])
                else: model=global_model
                tid=np.where(st==stage)[0]; fid=np.where(sf==stage)[0]
                tb[tid,ch,qi]=model.predict(ct[tid]); fb[fid,ch,qi]=model.predict(cf[fid])
    for b in (tb,fb):
        b[:,:,1]=np.maximum(b[:,:,1],b[:,:,0]); b[:,:,2]=np.maximum(b[:,:,2],b[:,:,1])
    return interval_exceedance(rtest[:,:2],tb), interval_exceedance(rfit[:,:2],fb)


def normalize_score(test,normal_reference):
    """Use only normal calibration scores to put both components on one scale."""
    med=np.median(normal_reference); mad=np.median(np.abs(normal_reference-med)); return np.maximum((test-med)/(1.4826*mad+.05),0)


def fuse_scores(environment_z, mahalanobis_z, environment_weight):
    """THIS IS THE COMBINATION: environment + double-device relation."""
    return environment_weight*environment_z + (1.0-environment_weight)*mahalanobis_z


def evaluate(score,y):
    auc=roc_auc_score(y,score); ap=average_precision_score(y,score); p,r,t=precision_recall_curve(y,score); f=2*p*r/(p+r+1e-9); i=int(np.nanargmax(f)); th=float(t[min(i,len(t)-1)]) if len(t) else float('inf'); pred=(score>=th).astype(int)
    return {'auroc':float(auc),'auprc':float(ap),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'normal_false_alarm':float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum()))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    x,meta=load(args.input,6); stable,info=stable_mask(x,meta); folds=[]; rows=[]
    for date in sorted(meta.date.unique()):
        ids=np.where(meta.date.values==date)[0]; normal=ids[(meta.error.values[ids]==0)&stable[ids]]; normal=normal[np.argsort(meta.start.values[normal])]; fit=normal[:max(8,int(len(normal)*.7))]; y=(meta.error.values[ids]!=0).astype(int)
        rfit,rtest,_=residuals(x[fit],x[ids]); edges=np.quantile(environment_conditions(x[fit])[:,1],[1/3,2/3])
        qtest,qfit=fit_quantile_score(x[fit],rfit,x[ids],rtest,edges); mtest=mahal(rfit,rtest); mfit=mahal(rfit,rfit)
        qz,mz=normalize_score(qtest,qfit),normalize_score(mtest,mfit)
        fold={'date':date,'fit_stable_normal':int(len(fit)),'test_windows':int(len(ids)),'positive_test':int(y.sum()),'humidity_edges':edges.tolist()}
        for w in (.25,.5,.75): fold[f'combined_w{int(w*100)}']=evaluate(fuse_scores(qz,mz,w),y)
        fold['environment_only']=evaluate(qtest,y); fold['mahalanobis_only']=evaluate(mtest,y); folds.append(fold)
        for i,idx in enumerate(ids):
            rows.append({'date':date,'window_index':int(idx),'error':int(meta.error.values[idx]),'environment_score':float(qtest[i]),'environment_score_normalized':float(qz[i]),'mahalanobis_score':float(mtest[i]),'mahalanobis_score_normalized':float(mz[i]),'combined_w25':float(fuse_scores(qz[i],mz[i],.25)),'combined_w50':float(fuse_scores(qz[i],mz[i],.50)),'combined_w75':float(fuse_scores(qz[i],mz[i],.75))})
    methods=['combined_w25','combined_w50','combined_w75','environment_only','mahalanobis_only']; summary={m:{k:float(np.mean([f[m][k] for f in folds])) for k in ['auroc','auprc','balanced_accuracy','precision','recall','normal_false_alarm']} for m in methods}
    payload={'protocol':{'target':'error==0 vs error!=0','environment_score':'P10/P50/P90 exceedance','joint_score':'Mahalanobis M4/M7/Mdiff/logratio','combination':'w*environment_z+(1-w)*mahalanobis_z','weights':[.25,.5,.75],'stable_info':info},'folds':folds,'summary':summary}; (args.output/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); pd.DataFrame(rows).to_csv(args.output/'window_scores.csv',index=False)
    report=['# P10/P90 + Mahalanobis 组合复测','','组合代码：`final = w * environment_z + (1-w) * mahalanobis_z`。','', '| 方法 | AUROC | AUPRC | 平衡准确率 | 精确率 | 召回率 | 正常误报率 |','|---|---:|---:|---:|---:|---:|---:|']
    for m,v in summary.items(): report.append(f"| {m} | {v['auroc']:.4f} | {v['auprc']:.4f} | {v['balanced_accuracy']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['normal_false_alarm']:.4f} |")
    report += ['', '`window_scores.csv` 保存每个窗口的环境分数、Mahalanobis分数、归一化分数和最终组合分数。']; (args.output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
