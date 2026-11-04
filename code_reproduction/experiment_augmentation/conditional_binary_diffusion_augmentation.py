"""Large binary conditional DDPM augmentation for experiment.xlsx.

This script deliberately generates both classes:
  normal (error == 0) and abnormal (error != 0).
The generated samples are evaluated only on held-out real experiment windows.
It is a local adapter, not an official upstream-paper implementation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score

from run_experiment_augmentation import (
    FEATURES_SHARED_ENV, fit_model, load_windows, one_hot,
    features_for_classifier, sample_model
)


def environment_condition(windows, meta, dates, env_mean=None, env_std=None):
    last=windows[:,-1,:]; first=windows[:,0,:]
    env=np.column_stack([last[:,2],last[:,3],(last[:,3]-first[:,3])/5.0,(last[:,2]-first[:,2])/5.0]).astype(np.float32)
    if env_mean is None: env_mean=env.mean(0)
    if env_std is None: env_std=env.std(0).clip(.05)
    return np.concatenate([one_hot(meta,dates,'binary'),(env-env_mean)/env_std],1).astype(np.float32),env_mean,env_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--normal-multiplier', type=int, default=4)
    ap.add_argument('--abnormal-multiplier', type=int, default=20)
    ap.add_argument('--feature-mode', choices=['shared_env'], default='shared_env')
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    features = FEATURES_SHARED_ENV
    x, meta = load_windows(args.input, 6, features)
    dates = sorted(meta.date.unique()); rng = np.random.default_rng(args.seed)
    train, test = [], []
    for (d, e), g in meta.groupby(['date', 'error'], sort=True):
        ids = g.index.to_numpy(); cut=max(1, int(len(ids)*.8))
        if len(ids) > 1: cut=min(cut, len(ids)-1)
        train.extend(ids[:cut]); test.extend(ids[cut:])
    train=np.asarray(train); test=np.asarray(test)
    xtr, xt = x[train], x[test]
    mtr, mt = meta.iloc[train].reset_index(drop=True), meta.iloc[test].reset_index(drop=True)
    cond, env_mean, env_std = environment_condition(xtr, mtr, dates)
    outputs = {'protocol': {'source': str(args.input), 'window_length': 6,
                            'features': features, 'condition': ['abnormal_flag', 'experiment_date','Temp_shared','Hum_shared','HumidityTrend','TempTrend'],
                            'train_windows': int(len(train)), 'test_windows': int(len(test)),
                            'real_train_normal': int((mtr.error == 0).sum()),
                            'real_train_abnormal': int((mtr.error != 0).sum()),
                            'normal_multiplier': args.normal_multiplier,
                            'abnormal_multiplier': args.abnormal_multiplier,
                            'evaluation': 'held-out real experiment windows; labels are original error values'}}

    for name, normalize in [('DDPM_raw_large', False), ('NsDiff_style_DDPM_large', True)]:
        model, mu, sd, hist = fit_model(xtr, cond, 20, args.epochs, 8e-4, args.seed, normalize)
        parts=[]; mparts=[]
        for abnormal, multiplier in [(0, args.normal_multiplier), (1, args.abnormal_multiplier)]:
            ids=np.where((mtr.error.values != 0) == bool(abnormal))[0]
            target=max(1, len(ids)*multiplier)
            chosen=rng.choice(ids, size=target, replace=True)
            c,_,_=environment_condition(xtr[chosen],mtr.iloc[chosen],dates,env_mean,env_std)
            gen=sample_model(model, c, mu, sd, 1, args.seed+abnormal+17)[0].reshape(target, 6, len(features))
            parts.append(gen); mparts.append(pd.DataFrame({'date':mtr.iloc[chosen].date.values,
                                                             'error':np.where(abnormal, 1, 0),
                                                             'synthetic':True}))
        syn=np.concatenate(parts); ms=pd.concat(mparts, ignore_index=True)
        ytr=(mtr.error.values != 0).astype(int); yte=(mt.error.values != 0).astype(int)
        real_feat=features_for_classifier(xtr); test_feat=features_for_classifier(xt); syn_feat=features_for_classifier(syn)
        clf=RandomForestClassifier(n_estimators=400,random_state=args.seed,class_weight='balanced_subsample',min_samples_leaf=2)
        clf.fit(real_feat,ytr); base=clf.predict(test_feat); base_prob=clf.predict_proba(test_feat)[:,1]
        clf2=RandomForestClassifier(n_estimators=400,random_state=args.seed,class_weight='balanced_subsample',min_samples_leaf=2)
        syn_y=ms.error.values.astype(int); clf2.fit(np.vstack([real_feat,syn_feat]), np.r_[ytr,syn_y])
        aug=clf2.predict(test_feat); aug_prob=clf2.predict_proba(test_feat)[:,1]
        outputs[name]={'epochs':args.epochs,'synthetic_windows':int(len(syn)),
                       'synthetic_normal':int((ms.error==0).sum()),'synthetic_abnormal':int((ms.error!=0).sum()),
                       'baseline_balanced_accuracy':float(balanced_accuracy_score(yte,base)),
                       'augmented_balanced_accuracy':float(balanced_accuracy_score(yte,aug)),
                       'baseline_macro_f1':float(f1_score(yte,base,average='macro',zero_division=0)),
                       'augmented_macro_f1':float(f1_score(yte,aug,average='macro',zero_division=0)),
                       'baseline_auroc':float(roc_auc_score(yte,base_prob)),
                       'augmented_auroc':float(roc_auc_score(yte,aug_prob)),
                       'baseline_auprc':float(average_precision_score(yte,base_prob)),
                       'augmented_auprc':float(average_precision_score(yte,aug_prob)),
                       'final_loss':float(hist[-1])}
        pd.DataFrame(syn.reshape(len(syn),-1)).to_csv(args.output/(name+'_synthetic_windows.csv'),index=False)
        ms.to_csv(args.output/(name+'_synthetic_metadata.csv'),index=False)
    (args.output/'results.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# experiment.xlsx 双类别条件扩散大规模增强','','本实验同时生成 error=0 正常样本和 error!=0 异常样本，测试只使用未参与生成训练的真实窗口。','', '| 方法 | 合成正常 | 合成异常 | 真实基线BA | 增强后BA | 真实基线AUROC | 增强后AUROC |','|---|---:|---:|---:|---:|---:|---:|']
    for n in ['DDPM_raw_large','NsDiff_style_DDPM_large']:
        v=outputs[n]; lines.append(f"| {n} | {v['synthetic_normal']} | {v['synthetic_abnormal']} | {v['baseline_balanced_accuracy']:.4f} | {v['augmented_balanced_accuracy']:.4f} | {v['baseline_auroc']:.4f} | {v['augmented_auroc']:.4f} |")
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8'); print(json.dumps(outputs,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
