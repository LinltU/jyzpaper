"""Actual upstream diffusion training on isolated experiment.xlsx splits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'dataset' / 'experiment.xlsx'
DEFAULT_OUT = ROOT / 'experiment_augmentation' / 'runs' / 'upstream_large_20260910'
FEATURES = ['M4_residual', 'M7_residual', 'M4-M7', 'M4/M7', 'log(M4/M7)']
METHODS = ['CSDI', 'MTSCI', 'IMDiffusion', 'Diffusion-TS', 'D3R']
LENGTH = 4


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out):
    """Split raw rows before windowing, with a separate per-date normal calibration."""
    sheets = pd.read_excel(SOURCE, sheet_name=None)
    paired, audits, windows, metadata = [], [], [], []
    baselines = {}
    for date in sorted({s[:6] for s in sheets}):
        parts = []
        for pos in (4, 7):
            name = next(s for s in sheets if s[:6] == date and f'({pos})' in s)
            q = sheets[name].copy()
            q['row'] = np.arange(len(q)) + 2
            q['dt'] = pd.to_datetime(q.time, format='%Y-%m-%d_%H:%M:%S')
            q = q.sort_values('dt')
            parts.append(q.rename(columns={c: f'{c}{pos}' for c in q.columns if c != 'dt'}))
        a, b = parts
        b['dt7'] = b.dt
        q = pd.merge_asof(a, b, on='dt', tolerance=pd.Timedelta('40s'), direction='nearest')
        q = q.dropna(subset=['Measure7']).copy()
        q['lag_seconds'] = (q.dt-q.dt7).dt.total_seconds().abs()
        q = q.sort_values('lag_seconds').drop_duplicates('row7').sort_values('dt').reset_index(drop=True)
        q['agree'] = q.error4 == q.error7
        q['phase'] = ((q.error4 != q.error4.shift()) | (q.dt.diff().dt.total_seconds() > 180)).cumsum()
        q['date'] = date
        q['kept'] = False
        q['role'] = 'excluded'
        normal = q[(q.error4 == 0) & q.agree]
        # A conservative suffix near terminal normal level; never mix later fault stages.
        tail = normal.tail(40)
        terminal = tail[['Measure4', 'Measure7']].tail(10).median().to_numpy()
        relative = np.abs(tail[['Measure4', 'Measure7']].to_numpy()/terminal-1).max(axis=1)
        bad = np.flatnonzero(relative > .15)
        stable = tail.iloc[(bad[-1]+1 if len(bad) else 0):]
        if len(stable) < 22:
            raise ValueError(f'{date}: only {len(stable)} stable normals, need 22 for honest separation')
        calib = stable.iloc[:8]
        base = calib[['Measure4', 'Measure7']].median().to_numpy()
        baselines[date] = base.tolist()
        q.loc[calib.index, 'role'] = 'calibration'
        eligible = stable.iloc[8:].index.to_list()
        for phase, g in q[q.error4 != 0].groupby('phase'):
            # Late phase sampling is prespecified; no exclusion based on abnormal amplitude.
            retained = g[g.agree].iloc[2:]
            eligible.extend(retained.index.to_list())
        q.loc[eligible, 'kept'] = True
        for phase, g in q.groupby('phase'):
            valid = g[g.kept & g.agree]
            # Separate contiguous stretches so dropped observations do not bridge windows.
            for _, segment in valid.groupby(valid.index.to_series().diff().ne(1).cumsum()):
                n = len(segment)
                # Chronological train/validation split. Tiny phases are training-only.
                cut = int(n*.65) if n >= 2*LENGTH else n
                for role, sub in [('fit', segment.iloc[:cut]), ('validation', segment.iloc[cut:])]:
                    q.loc[sub.index, 'role'] = role
                    for start in range(0, len(sub)-LENGTH+1):
                        block = sub.iloc[start:start+LENGTH]
                        m4, m7 = block.Measure4.to_numpy(), block.Measure7.to_numpy()
                        x = np.stack([m4-base[0], m7-base[1], m4-m7, m4/m7, np.log(m4/m7)], axis=-1)
                        windows.append(x)
                        metadata.append({'date': date, 'phase': int(phase), 'role': role,
                                         'label': int(block.error4.iloc[0] != 0), 'error': int(block.error4.iloc[0]),
                                         'start': str(block.dt.iloc[0]), 'end': str(block.dt.iloc[-1]),
                                         'row4': block.row4.astype(int).tolist(), 'row7': block.row7.astype(int).tolist(),
                                         'base4': float(base[0]), 'base7': float(base[1])})
            audits.append({'date': date, 'phase': int(phase), 'error': int(g.error4.iloc[0]),
                           'matched_rows': len(g), 'kept_rows': len(valid),
                           'normal_stable_start': str(stable.dt.iloc[0]) if int(g.error4.iloc[0]) == 0 else None})
        paired.append(q)
    x = np.stack(windows).astype(np.float32)
    meta = pd.DataFrame(metadata)
    # With very short anomalies, validation must retain entire held-out phases instead.
    # Move the last abnormal phase's windows in each date out of fitting.
    for date in meta.date.unique():
        ids = meta.index[(meta.date == date) & (meta.label == 1)]
        phases = meta.loc[ids, 'phase'].unique()
        if len(phases) >= 2:
            meta.loc[ids[meta.loc[ids, 'phase'].to_numpy() == phases[-1]], 'role'] = 'validation'
    np.savez_compressed(out/'dataset.npz', x=x)
    save_json(out/'metadata.json', meta.to_dict('records'))
    pd.concat(paired).to_csv(out/'paired_row_audit.csv', index=False, encoding='utf-8-sig')
    save_json(out/'phase_audit.json', audits)
    protocol = {'source': str(SOURCE), 'sha256': sha(SOURCE), 'raw_rows': sum(map(len,sheets.values())),
                'features': FEATURES, 'window_length': LENGTH, 'windows': len(x), 'baselines': baselines,
                'normal_rule': 'last <=40 normal rows; contiguous suffix within 15% of terminal 10-point median in both devices',
                'abnormal_rule': 'discard first 2 matched points per physical fault phase; do not filter by fault amplitude',
                'baseline': 'first 8 stable normal pairs per date, excluded from all model training and scoring',
                'pairing': 'nearest <=40s; one-to-one; exclude disagreeing error labels',
                'split': 'leave one date out; other dates chronological fit/validation, last fault phase validation; overlapping 4-step windows only within split and phase',
                'label': 'error != 0; supervised binary augmentation; no label as detector feature',
                'evaluation': 'no point adjustment; threshold selected only on validation; held-out date normal calibration required',
                'counts': meta.groupby(['date','role','label']).size().reset_index(name='n').to_dict('records')}
    save_json(out/'protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Upstream:
    def __init__(self, method, steps=30, hidden=32, device=None):
        self.method = method
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        p = ROOT/'upstream'/method
        config = {'diffusion': {'layers': 2, 'channels': hidden, 'nheads': 4,
                   'diffusion_embedding_dim': 64, 'beta_start': .0001, 'beta_end': .5,
                   'num_steps': steps, 'schedule': 'quad', 'is_linear': False, 'seqlen': LENGTH},
                  'model': {'is_unconditional': 0, 'timeemb': 32, 'featureemb': 8, 'target_strategy': 'random'}}
        self.config = config
        if method in ['CSDI','IMDiffusion']:
            load_module('diff_models', p/'diff_models.py')
            module = load_module(f'upstream_{method}', p/'main_model.py')
            self.model = module.CSDI_base(5, config, str(self.device))
        elif method == 'MTSCI':
            load_module('diff_block', p/'models'/'diff_block.py')
            module = load_module('upstream_MTSCI', p/'models'/'model.py')
            self.model = module.MTSCI_base(5, config, str(self.device))
        elif method == 'Diffusion-TS':
            sys.path.insert(0, str(p))
            from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
            self.config = dict(seq_length=LENGTH, feature_size=5, n_layer_enc=1, n_layer_dec=2,
                               d_model=hidden, timesteps=steps, sampling_timesteps=steps,
                               loss_type='l2', beta_schedule='cosine', n_heads=4,
                               mlp_hidden_times=2, use_ff=True)
            self.model = Diffusion_TS(**self.config).to(self.device)
            # The upstream log-frequency rule selects zero bins for four-point windows.
            # Retain its sole available frequency, preserving the Fourier seasonal branch.
            import types
            for layer in self.model.modules():
                if type(layer).__name__ == 'FourierLayer':
                    original = layer.topk_freq
                    def short_topk(self, xf, original=original):
                        if xf.shape[1] != 1:
                            return original(xf)
                        a,b = torch.meshgrid(torch.arange(xf.size(0)),torch.arange(xf.size(2)),indexing='ij')
                        ids = torch.zeros((xf.size(0),1,xf.size(2)),dtype=torch.long)
                        return xf,(a.unsqueeze(1),ids,b.unsqueeze(1))
                    layer.topk_freq = types.MethodType(short_topk,layer)
        elif method == 'D3R':
            sys.path.insert(0, str(p))
            from model.dddr import DDDR
            self.config = dict(time_steps=steps, beta_start=.0001, beta_end=.02,
                               window_size=LENGTH, model_dim=hidden, ff_dim=hidden*2,
                               atten_dim=hidden, feature_num=5, time_num=1, block_num=1,
                               head_num=4, dropout=.05, device=str(self.device), d=1, t=steps-1)
            self.model = DDDR(**self.config).to(self.device)
        else:
            raise ValueError(method)
        self.model = self.model.to(self.device)
        self.module_file = inspect.getfile(type(self.model))

    def mask_side(self, x):
        x = x.to(self.device)
        obs = x.permute(0,2,1)
        mask = torch.zeros_like(obs)
        mask[:, :, :LENGTH//2] = 1
        tp = torch.arange(LENGTH, device=self.device).float().repeat(len(x),1)
        return obs, mask, self.model.get_side_info(tp,mask), tp

    def loss(self, x, next_x=None):
        x = x.to(self.device)
        if next_x is not None:
            next_x = next_x.to(self.device)
        if self.method == 'Diffusion-TS':
            return self.model(x)
        if self.method == 'D3R':
            tp = torch.linspace(0,1,LENGTH, device=self.device).view(1,LENGTH,1).repeat(len(x),1,1)
            stable, _, recon = self.model(x,tp,p=.05)
            target_stable = x-x.mean(dim=1,keepdim=True)
            return .5*((stable-target_stable)**2).mean()+.5*((recon-x)**2).mean()
        obs, mask, side, tp = self.mask_side(x)
        full = torch.ones_like(obs)
        if self.method == 'MTSCI':
            reverse = 1-mask
            noise, consistency = self.model.calc_loss(obs, mask, full, reverse,
                next_x.permute(0,2,1) if next_x is not None else None,
                side, self.model.get_side_info(tp,reverse),
                self.model.get_side_info(tp,full), self.model.get_side_info(tp,torch.zeros_like(obs)))
            return noise+5*consistency
        kwargs = {'strategy_type': torch.zeros(len(x), dtype=torch.long, device=self.device)} if self.method == 'IMDiffusion' else {}
        return self.model.calc_loss(obs,mask,full,side,1,**kwargs)

    @torch.no_grad()
    def sample(self, anchors):
        anchors = anchors.to(self.device)
        self.model.eval()
        if self.method == 'Diffusion-TS':
            return self.model.sample(anchors.shape)
        if self.method == 'D3R':
            tp = torch.linspace(0,1,LENGTH, device=self.device).view(1,LENGTH,1).repeat(len(anchors),1,1)
            return self.model(anchors,tp,p=0)[2]
        obs, mask, side, _ = self.mask_side(anchors)
        kwargs = {'strategy_type': torch.zeros(len(anchors), dtype=torch.long, device=self.device)} if self.method == 'IMDiffusion' else {}
        samples = self.model.impute(obs,mask,side,1,**kwargs)[:,0]
        return (mask*obs+(1-mask)*samples).permute(0,2,1)


def scores(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {'Precision': float(precision_score(y,pred,zero_division=0)),
            'Recall': float(recall_score(y,pred,zero_division=0)),
            'F1': float(f1_score(y,pred,zero_division=0)),
            'ROC-AUC': float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else None,
            'Average Precision': float(average_precision_score(y,prob)) if y.sum() else None}


def threshold_from_validation(y, p):
    candidates = np.unique(np.r_[p, .5, np.nextafter(p.max(), np.inf)])
    # Prefer higher threshold on ties; no test labels participate.
    return float(max(candidates, key=lambda t:(f1_score(y,p>=t,zero_division=0),t)))


def classifier_features(x):
    return np.concatenate([x[:,-1],x.mean(1),x.std(1),x[:,-1]-x[:,0]],axis=1)


def train_classifier(name, xt, yt, xv, yv, xe, ye, seed, weight=None):
    if name == 'RF':
        model = RandomForestClassifier(n_estimators=160,max_depth=8,min_samples_leaf=2,
                                       random_state=seed,n_jobs=2,class_weight='balanced')
    else:
        model = HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=5,
                                               l2_regularization=1,random_state=seed)
    model.fit(classifier_features(xt),yt,sample_weight=weight)
    vp = model.predict_proba(classifier_features(xv))[:,1]
    ep = model.predict_proba(classifier_features(xe))[:,1]
    threshold = threshold_from_validation(yv,vp)
    return model,scores(ye,ep,threshold),scores(yv,vp,threshold),ep,threshold


def run(args):
    out = args.output
    run_dir = out/args.method/f'seed{args.seed}'/args.test_date
    run_dir.mkdir(parents=True,exist_ok=True)
    started = time.time()
    seed_all(args.seed)
    x = np.load(out/'dataset.npz')['x']
    meta = pd.read_json(out/'metadata.json',dtype={'date':str})
    y = meta.label.to_numpy()
    fit = np.flatnonzero((meta.date != args.test_date) & (meta.role == 'fit'))
    val = np.flatnonzero((meta.date != args.test_date) & (meta.role == 'validation'))
    test = np.flatnonzero(meta.date == args.test_date)
    assert not (set(fit)&set(val) or set(fit)&set(test) or set(val)&set(test))
    for name,ids in [('fit',fit),('validation',val),('test',test)]:
        if len(np.unique(y[ids]))!=2:
            raise ValueError(f'{name} must have both classes: {y[ids]}')
    scaler = MinMaxScaler(feature_range=(-.8,.8)).fit(x[fit].reshape(-1,5))
    z = scaler.transform(x.reshape(-1,5)).reshape(x.shape).astype(np.float32)
    histories, source_files, generated, generated_y, anchor_ids = [],{},[],[],[]
    # Train independent class-conditional generators; error is never a detection input.
    for label in (0,1):
        ids = fit[y[fit]==label]
        model = Upstream(args.method,args.steps,args.hidden)
        source_files[model.module_file] = sha(model.module_file)
        # Record every imported upstream Python file, including denoisers and attention.
        for mod in list(sys.modules.values()):
            f = getattr(mod,'__file__',None)
            if f and str(ROOT/'upstream'/args.method) in str(f) and str(f).endswith('.py'):
                source_files[str(f)] = sha(f)
        optimizer = torch.optim.Adam(model.model.parameters(),lr=.001)
        hist = []
        inp = torch.from_numpy(z[ids])
        for epoch in range(args.epochs):
            model.model.train()
            order = torch.randperm(len(inp))
            losses = []
            for start in range(0,len(inp),32):
                selected = order[start:start+32]
                loss = model.loss(inp[selected])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(),1)
                optimizer.step()
                losses.append(float(loss.detach()))
            hist.append(float(np.mean(losses)))
            if epoch in (0,args.epochs-1) or (epoch+1)%20==0:
                print(f'{args.method} {args.test_date} seed={args.seed} class={label} epoch={epoch+1} loss={hist[-1]:.6f}',flush=True)
        torch.save({'state_dict':model.model.state_dict(),'config':model.config,
                    'scaler_min':scaler.min_,'scaler_scale':scaler.scale_,'fit_ids':ids,
                    'features':FEATURES,'epochs':args.epochs},run_dir/f'class{label}.pt')
        histories.append({'label':label,'n_train':len(ids),'loss':hist,'parameters':sum(p.numel() for p in model.model.parameters()),
                          'config':model.config,'upstream_class':type(model.model).__name__})
        count = args.generate//2 if args.smoke else max(args.generate//2, 100*len(fit)//2)
        rng = np.random.default_rng(args.seed+label)
        chosen = rng.choice(ids,count,replace=True)
        chunks=[]
        for start in range(0,count,args.sample_batch):
            a = chosen[start:start+args.sample_batch]
            s = model.sample(torch.from_numpy(z[a])).numpy()
            chunks.append(s)
            if start==0 or start+args.sample_batch>=count or start%(args.sample_batch*10)==0:
                print(f'{args.method} class={label} generated {min(start+args.sample_batch,count)}/{count}',flush=True)
        gen = np.concatenate(chunks)
        generated.append(scaler.inverse_transform(gen.reshape(-1,5)).reshape(gen.shape).astype(np.float32))
        generated_y.extend([label]*len(gen))
        anchor_ids.extend(chosen.tolist())
    synthetic = np.concatenate(generated)
    sy = np.asarray(generated_y)
    anchors = np.asarray(anchor_ids)
    finite = np.isfinite(synthetic).all(axis=(1,2))
    if not finite.all():
        raise ValueError(f'Nonfinite synthetic windows: {(~finite).sum()}')
    base = meta[['base4','base7']].to_numpy()[anchors]
    m4 = synthetic[:,:,0]+base[:,None,0]
    m7 = synthetic[:,:,1]+base[:,None,1]
    positive = ((m4>0)&(m7>0)).all(axis=1)
    corrected = synthetic.copy()
    m4 = np.maximum(m4,1e-6)
    m7 = np.maximum(m7,1e-6)
    corrected[:,:,0] = m4-base[:,None,0]
    corrected[:,:,1] = m7-base[:,None,1]
    corrected[:,:,2] = m4-m7
    corrected[:,:,3] = m4/m7
    corrected[:,:,4] = np.log(m4/m7)
    # Relationship projection and validity filtering are evaluated separately from raw generation.
    np.savez_compressed(run_dir/'synthetic.npz',raw=synthetic,projected=corrected,label=sy,anchor_ids=anchors,positive=positive)
    save_json(run_dir/'training_history.json',histories)
    rows=[]
    rng=np.random.default_rng(args.seed)
    order=np.ravel(np.column_stack([rng.permutation(np.flatnonzero(sy==0)),rng.permutation(np.flatnonzero(sy==1))]))
    for clf in ('RF','HGB'):
        _,ev,vv,prob,threshold=train_classifier(clf,x[fit],y[fit],x[val],y[val],x[test],y[test],args.seed)
        rows.append({'method':'Real-only','classifier':clf,'regime':'real','n_synthetic':0,'metrics':ev,'validation':vv,'threshold':threshold})
        pd.DataFrame({'test_id':test,'label':y[test],'score':prob,'threshold':threshold}).to_csv(run_dir/f'{clf}_real_predictions.csv',index=False)
        regimes=[('1x',len(fit)),('10x',10*len(fit)),('100x',100*len(fit)),('large',len(synthetic))]
        for mode,pool in [('raw',synthetic),('projected',corrected)]:
            valid_order = order if mode=='raw' else order[positive[order]]
            for regime,n in regimes:
                sel=valid_order[:n]
                if not len(sel):
                    continue
                xx=np.concatenate([x[fit],pool[sel]])
                yy=np.r_[y[fit],sy[sel]]
                # Keep effective synthetic weight equal to real weight as volume increases.
                weights=np.r_[np.ones(len(fit)),np.full(len(sel),len(fit)/len(sel))]
                _,ev,vv,prob,threshold=train_classifier(clf,xx,yy,x[val],y[val],x[test],y[test],args.seed,weights)
                tag=f'{clf}_{mode}_{regime}'
                rows.append({'method':args.method,'classifier':clf,'regime':f'{mode}_{regime}',
                             'n_synthetic':len(sel),'metrics':ev,'validation':vv,'threshold':threshold})
                pd.DataFrame({'test_id':test,'label':y[test],'score':prob,'threshold':threshold}).to_csv(run_dir/f'{tag}_predictions.csv',index=False)
    result={'method':args.method,'seed':args.seed,'test_date':args.test_date,'status':'completed',
            'fit_ids':fit.tolist(),'val_ids':val.tolist(),'test_ids':test.tolist(),'features':FEATURES,
            'upstream_sources':source_files,'generated_windows':len(synthetic),'positive_windows':int(positive.sum()),
            'generation_std':synthetic.std(axis=(0,1)).tolist(),'real_std':x[fit].std(axis=(0,1)).tolist(),
            'training':histories,'results':rows,'elapsed_seconds':time.time()-started,
            'command':sys.argv,'input_sha256':sha(SOURCE)}
    save_json(run_dir/'result.json',result)
    print(f'COMPLETED {args.method} {args.test_date} seed {args.seed}: {len(synthetic)} generated, {len(rows)} evaluations, {time.time()-started:.1f}s',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--method',choices=METHODS)
    ap.add_argument('--test-date',default='250718')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--steps',type=int,default=30)
    ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--generate',type=int,default=10000)
    ap.add_argument('--sample-batch',type=int,default=256)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        prepare(args.output)
    else:
        run(args)


if __name__=='__main__':
    main()
