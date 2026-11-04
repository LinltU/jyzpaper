"""Small-sample multi-expert benchmark for experiment.xlsx.

Architecture implemented here:

    shared environment expert: global P10/P50/P90 residual bands
    low/high humidity experts: stage-specific quantile bands
    relation expert: stage-specific Mahalanobis on [M4,M7,Mdiff,logratio]
    soft gate: humidity distance to low/high calibration centers

This is an interpretable local multi-expert baseline. It is intentionally
smaller than a neural mixture-of-experts because each date has few stable
normal windows.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from stable_residual_diffusion import load, stable_mask, residuals, mahal


def conditions(w):
    last, first = w[:, -1, :], w[:, 0, :]; d=max(1,w.shape[1]-1)
    return np.c_[last[:,2],last[:,3],(last[:,2]-first[:,2])/d,(last[:,3]-first[:,3])/d,last[:,7]]


def interval_score(values, bands):
    out=[]
    for j in range(2):
        lo,hi=bands[:,j,0],bands[:,j,2]; width=np.maximum(hi-lo,.03)
        out.append(np.maximum((lo-values[:,j])/width,0)+np.maximum((values[:,j]-hi)/width,0))
    return np.maximum(out[0],out[1])


def fit_stage_quantiles(xfit,rfit,xtest,rtest,stage_fit,stage_test,global_model=True):
    """Fit shared and stage residual quantile experts for M4/M7."""
    cf,ct=conditions(xfit),conditions(xtest); tb=np.zeros((len(xtest),2,3)); fb=np.zeros((len(xfit),2,3))
    for j in range(2):
        for qi,alpha in enumerate((.1,.5,.9)):
            shared=GradientBoostingRegressor(loss='quantile',alpha=alpha,n_estimators=80,max_depth=2,learning_rate=.04,random_state=42).fit(cf,rfit[:,j])
            for stage in [0,1]:
                ids=np.where(stage_fit==stage)[0]
                if len(ids)>=12:
                    expert=GradientBoostingRegressor(loss='quantile',alpha=alpha,n_estimators=60,max_depth=2,learning_rate=.04,random_state=42).fit(cf[ids],rfit[ids,j])
                else: expert=shared
                tid=np.where(stage_test==stage)[0]; fid=np.where(stage_fit==stage)[0]
                tb[tid,j,qi]=expert.predict(ct[tid]); fb[fid,j,qi]=expert.predict(cf[fid])
            # Fill missing stage arrays using shared predictions before gating.
            miss=np.where(stage_test<0)[0]
            if len(miss): tb[miss,j,qi]=shared.predict(ct[miss])
    for b in (tb,fb): b[:,:,1]=np.maximum(b[:,:,1],b[:,:,0]); b[:,:,2]=np.maximum(b[:,:,2],b[:,:,1])
    return tb,fb


def soft_gate(humidity, low_center, high_center):
    """Return soft low/high humidity expert weights, not a hard switch."""
    scale=max(abs(high_center-low_center),1.0)
    dlow=((humidity-low_center)/scale)**2; dhigh=((humidity-high_center)/scale)**2
    logits=np.c_[-dlow,-dhigh]; logits-=logits.max(1,keepdims=True); w=np.exp(logits); return w/w.sum(1,keepdims=True)


def robust_z(test,normal):
    med=np.median(normal); mad=np.median(np.abs(normal-med)); return np.maximum((test-med)/(1.4826*mad+.05),0)


def evaluate(s,y):
    auc=roc_auc_score(y,s); ap=average_precision_score(y,s); p,r,t=precision_recall_curve(y,s); f=2*p*r/(p+r+1e-9); i=int(np.nanargmax(f)); th=float(t[min(i,len(t)-1)]) if len(t) else float('inf'); pred=(s>=th).astype(int)
    return {'auroc':float(auc),'auprc':float(ap),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'normal_false_alarm':float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum()))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    x,meta=load(args.input,6); stable,info=stable_mask(x,meta); folds=[]; rows=[]
    for date in sorted(meta.date.unique()):
        ids=np.where(meta.date.values==date)[0]; normal=ids[(meta.error.values[ids]==0)&stable[ids]]; normal=normal[np.argsort(meta.start.values[normal])]; fit=normal[:max(8,int(len(normal)*.7))]; y=(meta.error.values[ids]!=0).astype(int)
        rfit,rtest,_=residuals(x[fit],x[ids]); cfit,ctest=conditions(x[fit]),conditions(x[ids]); humidity_fit,humidity_test=cfit[:,1],ctest[:,1]
        threshold=float(np.median(humidity_fit)); stage_fit=(humidity_fit>=threshold).astype(int); stage_test=(humidity_test>=threshold).astype(int)
        low_center=float(np.median(humidity_fit[stage_fit==0])); high_center=float(np.median(humidity_fit[stage_fit==1])); gate=soft_gate(humidity_test,low_center,high_center)

        # Shared + stage-specific P10/P50/P90 experts.
        bands,fit_bands=fit_stage_quantiles(x[fit],rfit,x[ids],rtest,stage_fit,stage_test)
        env_stage=interval_score(rtest[:,:2],bands)
        env_shared=interval_score(rtest[:,:2],np.repeat(fit_bands.mean(0,keepdims=True),len(ids),axis=0))
        env_z=robust_z(env_stage,interval_score(rfit[:,:2],fit_bands))

        # Stage-specific relation experts.  If one stage is too small, use all
        # stable normal residuals as a safe shared fallback.
        relation_scores=[]; relation_fit_scores=[]
        for stage in [0,1]:
            train_ids=np.where(stage_fit==stage)[0]; train_ids=train_ids if len(train_ids)>=12 else np.arange(len(rfit)); test_ids=np.where(stage_test==stage)[0]
            s=np.zeros(len(ids)); sf=np.zeros(len(fit)); s[test_ids]=mahal(rfit[train_ids],rtest[test_ids]); sf[train_ids]=mahal(rfit[train_ids],rfit[train_ids]); relation_scores.append(s); relation_fit_scores.append(sf)
        relation=np.where(gate[:,0]>=gate[:,1],relation_scores[0],relation_scores[1]); relation_ref=np.concatenate([relation_fit_scores[0][relation_fit_scores[0]>0],relation_fit_scores[1][relation_fit_scores[1]>0]])
        relation_z=robust_z(relation,relation_ref if len(relation_ref) else mahal(rfit,rfit)); final=.25*env_z+.75*relation_z
        fold={'date':date,'fit_stable_normal':int(len(fit)),'test_windows':int(len(ids)),'positive_test':int(y.sum()),'humidity_threshold':threshold,'low_center':low_center,'high_center':high_center,'multi_expert':evaluate(final,y),'environment_expert':evaluate(env_stage,y),'relation_expert':evaluate(relation,y),'shared_mahalanobis':evaluate(mahal(rfit,rtest),y)}; folds.append(fold)
        for i,idx in enumerate(ids): rows.append({'date':date,'window_index':int(idx),'error':int(meta.error.values[idx]),'humidity_stage':int(stage_test[i]),'gate_low':float(gate[i,0]),'gate_high':float(gate[i,1]),'environment_score':float(env_stage[i]),'relation_score':float(relation[i]),'combined_score':float(final[i])})
    methods=['multi_expert','environment_expert','relation_expert','shared_mahalanobis']; summary={m:{k:float(np.mean([f[m][k] for f in folds])) for k in ['auroc','auprc','balanced_accuracy','precision','recall','normal_false_alarm']} for m in methods}; payload={'protocol':{'target':'error==0 vs error!=0','experts':['shared environment','low humidity','high humidity','stage relation Mahalanobis'],'gate':'soft humidity-distance gate','fusion':'0.25*environment + 0.75*relation','stable_info':info},'folds':folds,'summary':summary}; (args.output/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); pd.DataFrame(rows).to_csv(args.output/'window_scores.csv',index=False); (args.output/'REPORT.md').write_text('# 多专家环境/双设备关系复测\n\n'+json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
