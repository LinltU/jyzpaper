"""Environment-conditioned P10/P50/P90 and humidity-stage benchmark.

Only experiment.xlsx mirror data is used.  Normal stable windows are used to
fit conditional quantile bands for the final M4/M7 values.  error!=0 is only
used for offline evaluation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_curve, precision_score, recall_score
from stable_residual_diffusion import load, stable_mask, residuals, mahal


def cond(x):
    last=x[:,-1,:]; first=x[:,0,:]; den=max(1,x.shape[1]-1)
    return np.c_[last[:,2],last[:,3],(last[:,2]-first[:,2])/den,(last[:,3]-first[:,3])/den,last[:,7]]


def score_eval(score,y):
    auc=roc_auc_score(y,score); ap=average_precision_score(y,score); p,r,t=precision_recall_curve(y,score); f=2*p*r/(p+r+1e-9); i=int(np.nanargmax(f)); th=float(t[min(i,len(t)-1)]) if len(t) else float('inf'); pred=(score>=th).astype(int)
    return {'auroc':float(auc),'auprc':float(ap),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'normal_false_alarm':float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum())),'threshold':th}


def fit_quantile_band(xfit, rfit, xall, humidity_edges):
    cf=cond(xfit); ca=cond(xall); hf=cf[:,1]; ha=ca[:,1]
    stage_fit=np.digitize(hf,humidity_edges); stage_all=np.digitize(ha,humidity_edges)
    preds=np.zeros((len(xall),2,3),dtype=float)
    # Small samples benefit from a global model plus a stage-specific model
    # only when a stage has enough normal windows.
    for j,target_col in enumerate([0,1]):
        for qi,alpha in enumerate([.10,.50,.90]):
            global_model=GradientBoostingRegressor(loss='quantile',alpha=alpha,n_estimators=80,max_depth=2,learning_rate=.04,random_state=42)
            global_model.fit(cf,rfit[:,j])
            for stage in range(3):
                ids=np.where(stage_fit==stage)[0]
                use=ids if len(ids)>=12 else np.arange(len(cf))
                model=global_model if len(ids)<12 else GradientBoostingRegressor(loss='quantile',alpha=alpha,n_estimators=60,max_depth=2,learning_rate=.04,random_state=42)
                if model is not global_model: model.fit(cf[use],rfit[use,j])
                out_ids=np.where(stage_all==stage)[0]
                preds[out_ids,j,qi]=model.predict(ca[out_ids])
    # enforce non-crossing quantiles
    preds[:,:,1]=np.maximum(preds[:,:,1],preds[:,:,0]); preds[:,:,2]=np.maximum(preds[:,:,2],preds[:,:,1])
    return preds,stage_all


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    x,meta=load(args.input,6); stable,stable_info=stable_mask(x,meta); folds=[]
    for date in sorted(meta.date.unique()):
        ids=np.where(meta.date.values==date)[0]; normals=ids[(meta.error.values[ids]==0)&stable[ids]]; normals=normals[np.argsort(meta.start.values[normals])]; fit=normals[:max(8,int(len(normals)*.7))]; y=(meta.error.values[ids]!=0).astype(int)
        # Humidity stages are learned from the calibration normal data only.
        h=cond(x[fit])[:,1]; edges=np.quantile(h,[1/3,2/3]); cf=cond(x[fit]); ca=cond(x[ids]);
        # Regress M4/M7 after removing a simple environment relation; this
        # keeps the quantile bands focused on deviations, not absolute date.
        rfit,rtest,_=residuals(x[fit],x[ids]); band,stages=fit_quantile_band(x[fit],rfit,x[ids],edges)
        actual=rtest[:,:2]
        # compare residual targets to quantile bands learned on residuals
        s=[]
        for j in range(2):
            lo,med,hi=band[:,j,0],band[:,j,1],band[:,j,2]; scale=np.maximum(hi-lo,.03); s.append(np.maximum((lo-actual[:,j])/scale,0)+np.maximum((actual[:,j]-hi)/scale,0))
        score=np.max(np.c_[s[0],s[1]],axis=1)
        folds.append({'date':date,'fit_stable_normal':int(len(fit)),'test_windows':int(len(ids)),'positive_test':int(y.sum()),'humidity_edges':edges.tolist(),'humidity_stage_counts':np.bincount(stages,minlength=3).tolist(),'quantile_stage_score':score_eval(score,y)})
    summary={k:{q:float(np.mean([f['quantile_stage_score'][q] for f in folds])) for q in ['auroc','auprc','balanced_accuracy','precision','recall','normal_false_alarm']} for k in ['quantile_stage_score']}
    payload={'protocol':{'target':'error==0 vs error!=0','features':['M4','M7','Temp_shared','Hum_shared','TempTrend','HumidityTrend','elapsed_min'],'humidity_stage':'per-date calibration normal tertiles','quantiles':['P10','P50','P90'],'stable_info':stable_info},'folds':folds,'summary':summary}; (args.output/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
