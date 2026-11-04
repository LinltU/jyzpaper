"""Combine environment-conditioned P10/P50/P90 score with joint Mahalanobis.

Only experiment.xlsx mirror data is used.  The error label is used only for
offline evaluation; fitting uses stable error==0 windows per date.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_curve, precision_score, recall_score
from stable_residual_diffusion import load, stable_mask, residuals, mahal


def cond(x):
    last=x[:,-1,:]; first=x[:,0,:]; d=max(1,x.shape[1]-1)
    return np.c_[last[:,2],last[:,3],(last[:,2]-first[:,2])/d,(last[:,3]-first[:,3])/d,last[:,7]]


def quantile_score(xfit, rfit, xtest, rtest, edges):
    cf,ct=cond(xfit),cond(xtest); sf=np.digitize(cf[:,1],edges); st=np.digitize(ct[:,1],edges)
    bands=np.zeros((len(xtest),2,3)); fit_bands=np.zeros((len(xfit),2,3))
    for j in range(2):
        for qi,alpha in enumerate([.1,.5,.9]):
            global_model=GradientBoostingRegressor(loss='quantile',alpha=alpha,n_estimators=80,max_depth=2,learning_rate=.04,random_state=42).fit(cf,rfit[:,j])
            for stage in range(3):
                train_ids=np.where(sf==stage)[0]; use=train_ids if len(train_ids)>=12 else np.arange(len(cf))
                model=global_model if len(train_ids)<12 else GradientBoostingRegressor(loss='quantile',alpha=alpha,n_estimators=60,max_depth=2,learning_rate=.04,random_state=42).fit(cf[use],rfit[use,j])
                test_ids=np.where(st==stage)[0]; bands[test_ids,j,qi]=model.predict(ct[test_ids])
                fit_ids=np.where(sf==stage)[0]; fit_bands[fit_ids,j,qi]=model.predict(cf[fit_ids])
    for a in [bands,fit_bands]:
        a[:,:,1]=np.maximum(a[:,:,1],a[:,:,0]); a[:,:,2]=np.maximum(a[:,:,2],a[:,:,1])
    def out(v,b):
        z=[]
        for j in range(2):
            lo,hi=b[:,j,0],b[:,j,2]; scale=np.maximum(hi-lo,.03); z.append(np.maximum((lo-v[:,j])/scale,0)+np.maximum((v[:,j]-hi)/scale,0))
        return np.max(np.c_[z[0],z[1]],axis=1)
    return out(rtest[:,:2],bands),out(rfit[:,:2],fit_bands)


def robust_scale(test, ref):
    med=np.median(ref); mad=np.median(np.abs(ref-med)); return np.maximum((test-med)/(1.4826*mad+.05),0)


def evaluate(score,y):
    auc=roc_auc_score(y,score); ap=average_precision_score(y,score); p,r,t=precision_recall_curve(y,score); f=2*p*r/(p+r+1e-9); i=int(np.nanargmax(f)); th=float(t[min(i,len(t)-1)]) if len(t) else float('inf'); pred=(score>=th).astype(int)
    return {'auroc':float(auc),'auprc':float(ap),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'normal_false_alarm':float(((pred==1)&(y==0)).sum()/max(1,(y==0).sum())),'threshold':th}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    x,meta=load(args.input,6); stable,info=stable_mask(x,meta); folds=[]
    for date in sorted(meta.date.unique()):
        ids=np.where(meta.date.values==date)[0]; normals=ids[(meta.error.values[ids]==0)&stable[ids]]; normals=normals[np.argsort(meta.start.values[normals])]; fit=normals[:max(8,int(len(normals)*.7))]; y=(meta.error.values[ids]!=0).astype(int)
        rfit,rtest,_=residuals(x[fit],x[ids]); h=cond(x[fit])[:,1]; edges=np.quantile(h,[1/3,2/3]); qtest,qfit=quantile_score(x[fit],rfit,x[ids],rtest,edges); mtest=mahal(rfit,rtest); mfit=mahal(rfit,rfit)
        qz=robust_scale(qtest,qfit); mz=robust_scale(mtest,mfit); fold={'date':date,'fit_stable_normal':int(len(fit)),'test_windows':int(len(ids)),'positive_test':int(y.sum()),'humidity_edges':edges.tolist()}
        for w in [.25,.5,.75]: fold[f'combined_w{int(w*100)}']=evaluate(w*qz+(1-w)*mz,y)
        fold['environment_only']=evaluate(qtest,y); fold['mahalanobis_only']=evaluate(mtest,y); folds.append(fold)
    methods=['combined_w25','combined_w50','combined_w75','environment_only','mahalanobis_only']; summary={m:{k:float(np.mean([f[m][k] for f in folds])) for k in ['auroc','auprc','balanced_accuracy','precision','recall','normal_false_alarm']} for m in methods}; payload={'protocol':{'target':'error==0 vs error!=0','calibration':'per-date stable error==0 windows','environment_score':'P10/P90 exceedance of M4/M7 residuals','joint_score':'Mahalanobis of M4/M7/Mdiff/logratio residuals','weights_tested':[.25,.5,.75],'stable_info':info},'folds':folds,'summary':summary}; (args.output/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); (args.output/'REPORT.md').write_text('# 环境区间 + Mahalanobis 组合复测\n\n'+json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
