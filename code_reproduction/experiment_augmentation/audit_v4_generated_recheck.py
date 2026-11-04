"""Frozen-reference v4 source-context recheck, NOT standalone synthetic v4 detection.

Only Measure is injected, one sample at a time. Environment prediction, peers,
rank reference distributions and window robust scales are frozen from real data.
No generated evidence channel is trusted as an independently computed feature.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from plot_v4_generated_data import load_context, load_generated

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'v4_results_20261009'
DEFAULT_RUN = ROOT / '03_benchmark_v4_evidence_features'


def robust(x):
    med = np.nanmedian(x)
    return med, np.nanmedian(np.abs(x - med)) * 1.4826 + 1e-9


class RankReference:
    """Exact original average ranks at reference values; monotonic interpolation elsewhere.

    This explicit out-of-sample extension is not an original v4 fitted transform.
    Extreme values saturate at p=1e-4/1-1e-4, matching original clipping.
    """
    def __init__(self, values):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError('Empty rank reference')
        self.x, counts = np.unique(values, return_counts=True)
        self.p = (np.cumsum(counts) - (counts - 1) / 2) / len(values)

    def transform(self, x):
        return norm.ppf(np.clip(np.interp(x, self.x, self.p, left=1e-4, right=1-1e-4), 1e-4, 1-1e-4))


def rules(rmean, pmean, persist, q90, rcount, pcount, pmax, has_peer, rscale, pscale):
    rz = (np.abs(rmean) - rscale[0]) / rscale[1]
    pz = (np.abs(pmean) - pscale[0]) / pscale[1]
    consistent = (rz >= 3) & (persist >= .15) & (pz >= 3) & (np.sign(rmean) == np.sign(pmean))
    repeated = pcount >= 2
    extreme = pmax >= 12
    solo = ~has_peer & (persist >= .15) & (q90 >= 2) & (rcount >= 2)
    label = consistent | repeated | extreme | solo
    reason = np.select([consistent, repeated, extreme, solo], ['environment_and_peer', 'repeated_peer_jump', 'extreme_peer_jump', 'no_peer_residual'], default='normal')
    return label, reason


class Rechecker:
    def __init__(self, run):
        self.ctx = load_context(run)
        self.windows = pd.read_csv(run / 'v4_window_labels.csv', dtype={'IMEI': str})
        self.rscale = robust(np.abs(self.windows.residual_z_mean))
        self.pscale = robust(np.abs(self.windows.peer_z_mean))
        full = pd.concat(self.ctx.points.values(), ignore_index=True)
        env = {}
        assigned = np.zeros(len(full), bool)
        for key, ids in full.groupby('hum_bin', dropna=True).groups.items():
            if len(ids) >= 50:
                env[key] = RankReference(full.loc[ids, 'env_residual_z_device'])
                assigned[ids] = True
        efallback = RankReference(full.loc[~assigned, 'env_residual_z_device']) if (~assigned).any() else None
        peer = {}
        assigned = np.zeros(len(full), bool)
        valid = full.peer_delta.notna().to_numpy()
        for key, ids in full.loc[valid].groupby(['group_id', 'peer_hum_bin'], dropna=True).groups.items():
            if len(ids) >= 20:
                peer[key] = RankReference(full.loc[ids, 'peer_delta'])
                assigned[ids] = True
        pfallback = RankReference(full.loc[valid & ~assigned, 'peer_delta']) if (valid & ~assigned).any() else None
        self.devices = {}
        self.neutral = np.zeros(len(self.ctx.meta), dtype=int)
        for imei, p in self.ctx.points.items():
            m, s = robust(p.env_residual)
            _, ps = robust(p.peer_delta) if p.peer_delta.notna().any() else (0, 1)
            erefs = [env.get(key, efallback) for key in p.hum_bin]
            prefs = [peer.get((group, key), pfallback) if valid else None
                     for group, key, valid in zip(p.group_id, p.peer_hum_bin, p.peer_delta.notna())]
            self.devices[imei] = dict(p=p, med=m, scale=s, peer_scale=ps, erefs=erefs, prefs=prefs,
                                      prediction=p.Measure.to_numpy() - p.env_residual.to_numpy())
            # Round-trip tests: exact reference scores and original v4 labels.
            er = np.array([ref.transform(x) for ref, x in zip(erefs, p.env_residual_z_device)])
            np.testing.assert_allclose(er, p.env_residual_z, atol=1e-10)
            for ref, x, expected in zip(prefs, p.peer_delta, p.peer_delta_z):
                if ref is not None and np.isfinite(x):
                    np.testing.assert_allclose(ref.transform(x), expected, atol=1e-10)
            labels, _ = self.classify(imei, p.env_residual_z.to_numpy()[None], p.peer_delta_z.to_numpy()[None], p.peer_delta.to_numpy()[None])
            original = self.windows[self.windows.IMEI == imei].window_anomaly_v4.to_numpy()
            np.testing.assert_array_equal(labels[0], original)
        self.validate_real_windows()
        for imei, sub in self.ctx.meta.groupby('IMEI', sort=False):
            ids = sub.index.to_numpy()
            rows = np.asarray(sub.source_rows.to_list(), int)
            values = self.devices[imei]['prediction'][rows]
            self.neutral[ids] = self.evaluate_device(imei, rows, values)['rechecked_label']

    def classify(self, imei, rz, pz, pdelt):
        d = self.devices[imei]
        p = d['p']
        batch, n = rz.shape
        rjump = np.full_like(rz, np.nan)
        rjump[:, 1:] = np.abs(np.diff(rz, axis=1)) / np.sqrt(2)
        pjump = np.full_like(rz, np.nan)
        valid = np.flatnonzero(p.peer_delta.notna().to_numpy())
        complete = p.peer_complete.to_numpy(bool)
        if len(valid) > 1:
            eligible = complete[valid[1:]] & complete[valid[:-1]]
            cols = valid[1:][eligible]
            previous = valid[:-1][eligible]
            pjump[:, cols] = np.abs(pdelt[:, cols] - pdelt[:, previous]) / (np.sqrt(2) * d['peer_scale'])
        nw = n // 32 + int(n % 32 >= 2)
        results, reasons = [], []
        for w in range(nw):
            a, b = w * 32, min(n, (w + 1) * 32)
            r = rz[:, a:b]
            pp = pz[:, a:b]
            pj = pjump[:, a:b]
            count = np.isfinite(pp).sum(axis=1)
            pm = np.divide(np.nansum(pp, axis=1), count, out=np.full(batch, np.nan), where=count > 0)
            pmax = np.max(np.where(np.isfinite(pj), pj, -np.inf), axis=1)
            label, reason = rules(r.mean(axis=1), pm, (np.abs(r) >= 2).mean(axis=1),
                                  np.quantile(np.abs(r), .9, axis=1), (rjump[:, a:b] >= 2.5).sum(axis=1),
                                  (pj >= 8).sum(axis=1), pmax, count > 0, self.rscale, self.pscale)
            results.append(label)
            reasons.append(reason)
        return np.stack(results, axis=1), np.stack(reasons, axis=1)

    def evaluate_device(self, imei, rows, values):
        d = self.devices[imei]
        p = d['p']
        batch = len(rows)
        rr = np.broadcast_to(p.env_residual_z.to_numpy(), (batch, len(p))).copy()
        pp = np.broadcast_to(p.peer_delta_z.to_numpy(), rr.shape).copy()
        delta = np.broadcast_to(p.peer_delta.to_numpy(), rr.shape).copy()
        residual = values - d['prediction'][rows]
        delta_new = p.peer_delta.to_numpy()[rows] + values - p.Measure.to_numpy()[rows]
        rnew, pnew = np.empty_like(values), np.full_like(values, np.nan)
        # Group transform calls by source row to amortize rank lookup.
        for point in np.unique(rows):
            ii, jj = np.where(rows == point)
            rnew[ii, jj] = d['erefs'][point].transform((residual[ii, jj] - d['med']) / d['scale'])
            if d['prefs'][point] is not None:
                pnew[ii, jj] = d['prefs'][point].transform(delta_new[ii, jj])
        index = np.arange(batch)[:, None]
        rr[index, rows] = rnew
        pp[index, rows] = pnew
        delta[index, rows] = delta_new
        labs, reasons = self.classify(imei, rr, pp, delta)
        nw = labs.shape[1]
        touched = np.minimum(rows // 32, nw - 1)
        # A one-point tail is unlabelled under original MIN_TAIL_POINTS=2.
        eligible = rows < nw * 32
        selected = labs[index, touched] & eligible
        result = selected.any(axis=1)
        rtext = [';'.join(sorted(set(reasons[i, np.unique(touched[i][eligible[i]])][labs[i, np.unique(touched[i][eligible[i]])]]))) or 'normal'
                 for i in range(batch)]
        comp = p.peer_complete.to_numpy(bool)[rows]
        inside = np.abs(np.diff(delta_new, axis=1)) / (np.sqrt(2) * d['peer_scale'])
        inside = np.where(comp[:, 1:] & comp[:, :-1], inside, np.nan)
        maxjump = np.max(np.where(np.isfinite(inside), inside, -np.inf), axis=1)
        maxjump[~np.isfinite(maxjump)] = np.nan
        return dict(rechecked_label=result.astype(int), recheck_reason=rtext,
                    internal_large_peer_jumps=(inside >= 8).sum(axis=1), internal_peer_jump_max=maxjump,
                    internal_peer_trigger=((inside >= 8).sum(axis=1) >= 2) | (maxjump >= 12),
                    recalculated_rz=rnew, recalculated_pz=pnew,
                    touched_v4_windows=[';'.join(map(str, np.unique(t[ok]))) for t, ok in zip(touched, eligible)])

    def validate_real_windows(self):
        tested = 0
        for imei, sub in self.ctx.meta.groupby('IMEI', sort=False):
            ids = sub.index.to_numpy()
            rows = np.asarray(sub.source_rows.to_list(), int)
            out = self.evaluate_device(imei, rows, self.devices[imei]['p'].Measure.to_numpy()[rows])
            np.testing.assert_array_equal(out['rechecked_label'], sub.label.to_numpy())
            tested += len(ids)
        self.real_roundtrip_count = tested

    def file(self, path, ratio):
        values, labels, sources = load_generated(path, self.ctx, ratio)
        with np.load(path, allow_pickle=False) as z:
            channels = z['windows']
        frame = pd.DataFrame({'sample_id': np.arange(len(values)), 'source_id': sources,
                              'inherited_label': labels, 'IMEI': self.ctx.meta.IMEI.to_numpy()[sources],
                              'nonpositive_measure_points': (values <= 0).sum(axis=1),
                              'neutral_context_label': self.neutral[sources],
                              'standalone_v4_status': 'not_evaluable_4_points_no_synthetic_peer_panel'})
        for imei, sub in frame.groupby('IMEI', sort=False):
            ids = sub.index.to_numpy()
            rows = np.asarray(self.ctx.meta.iloc[sources[ids]].source_rows.to_list(), int)
            out = self.evaluate_device(imei, rows, values[ids])
            rz, pz = out.pop('recalculated_rz'), out.pop('recalculated_pz')
            for key, val in out.items():
                frame.loc[ids, key] = val
            frame.loc[ids, 'env_z_channel_mae'] = np.mean(np.abs(channels[ids, :, 1] - rz), axis=1)
            finite = np.isfinite(pz)
            denom = finite.sum(axis=1)
            frame.loc[ids, 'peer_z_channel_mae'] = np.divide(np.nansum(np.where(finite, np.abs(channels[ids, :, 2] - pz), np.nan), axis=1),
                                                           denom, out=np.full(len(ids), np.nan), where=denom > 0)
            frame.loc[ids, 'retained_points'] = np.isclose(values[ids], self.ctx.real[sources[ids], :, 0], rtol=1e-5, atol=1e-6).sum(axis=1)
        frame['label_agrees'] = frame.inherited_label == frame.rechecked_label
        frame['context_positive_even_when_neutral'] = (frame.neutral_context_label == 1) & (frame.rechecked_label == 1)
        frame['generated_induced_context_positive'] = (frame.neutral_context_label == 0) & (frame.rechecked_label == 1)
        for col in ['rechecked_label', 'internal_large_peer_jumps', 'retained_points']:
            frame[col] = frame[col].astype(int)
        frame['start_time'] = self.ctx.meta.start_time.to_numpy()[sources]
        frame['end_time'] = self.ctx.meta.end_time.to_numpy()[sources]
        return frame


def summary(frame):
    y, pred = frame.inherited_label.to_numpy(), frame.rechecked_label.to_numpy()
    tn, fp, fn, tp = [int(((y == a) & (pred == b)).sum()) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    pos = y == 1
    return dict(windows=len(frame), normal_to_normal=tn, normal_to_candidate=fp,
                positive_to_normal=fn, positive_to_candidate=tp,
                normal_agreement=tn / (tn + fp) if tn + fp else np.nan,
                positive_agreement=tp / (tp + fn) if tp + fn else np.nan,
                overall_agreement=(tn + tp) / len(frame),
                positive_neutral_context_candidates=int(frame.loc[pos, 'neutral_context_label'].sum()),
                positive_internal_peer_triggers=int(frame.loc[pos, 'internal_peer_trigger'].astype(bool).sum()),
                nonpositive_windows=int((frame.nonpositive_measure_points > 0).sum()),
                env_z_channel_mae=float(frame.env_z_channel_mae.mean()),
                peer_z_channel_mae=float(frame.peer_z_channel_mae.mean()))


def main(args):
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    checker = Rechecker(args.run)
    print(f'Real round-trip: {checker.real_roundtrip_count} windows passed', flush=True)
    configs, devices = [], []
    for ratio in args.ratios or checker.ctx.protocol['ratios']:
        for method in args.methods or checker.ctx.protocol['methods']:
            for seed in args.seeds or checker.ctx.protocol['seeds']:
                safe = method.replace('-', '_').replace('/', '_')
                path = args.run / f'raw_{ratio}_{safe}_seed{seed}.npz'
                frame = checker.file(path, ratio)
                frame.to_csv(args.output / f'rechecked_{ratio}_{safe}_seed{seed}.csv.gz', index=False, compression='gzip')
                identity = dict(ratio=ratio, method=method, seed=seed)
                configs.append(dict(**identity, **summary(frame), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                for imei, sub in frame.groupby('IMEI', sort=False):
                    devices.append(dict(**identity, IMEI=imei, **summary(sub)))
                print(f"{ratio} {method} seed={seed}: normal agreement={configs[-1]['normal_agreement']:.3%}; positive agreement={configs[-1]['positive_agreement']:.3%}", flush=True)
    table = pd.DataFrame(configs)
    table.to_csv(args.output / 'configuration_summary.csv', index=False)
    pd.DataFrame(devices).to_csv(args.output / 'device_summary.csv', index=False)
    pooled = table.groupby(['ratio', 'method'], sort=False)[['windows', 'normal_to_normal', 'normal_to_candidate', 'positive_to_normal', 'positive_to_candidate',
                                                            'positive_neutral_context_candidates', 'positive_internal_peer_triggers', 'nonpositive_windows']].sum().reset_index()
    pooled['normal_agreement'] = pooled.normal_to_normal / (pooled.normal_to_normal + pooled.normal_to_candidate)
    pooled['positive_agreement'] = pooled.positive_to_candidate / (pooled.positive_to_candidate + pooled.positive_to_normal)
    pooled.to_csv(args.output / 'method_summary.csv', index=False)
    protocol = dict(type='frozen-reference v4 source-context counterfactual audit', run=str(args.run),
                    configurations=len(table), generated_windows=int(table.windows.sum()), real_roundtrip_windows=checker.real_roundtrip_count,
                    window_reference_count=len(checker.windows), original_candidate_count=int(checker.windows.window_anomaly_v4.sum()),
                    elapsed_seconds=time.monotonic() - start,
                    thresholds=dict(window_z=3, persistence=.15, peer_jump=8, min_peer_jumps=2, isolated_peer_jump=12, residual_jump=2.5),
                    window_size=32, replacement='all four Measure points at exact source rows, one sample per independent scenario',
                    frozen='original environment prediction, residual median/MAD, observed peer residual median, peer MAD, rank distributions, window median/MAD',
                    rank_extension='piecewise-linear average-rank probability at original unique values, clipped to [1e-4,1-1e-4]; this is an audit extension of v4, not refitting v4',
                    scope='target-device labels only; peers fixed as observed; not a jointly synthesized fleet',
                    neutral_control='replace all four source Measure with frozen environment predicted Measure; NOT a guaranteed normal signal',
                    disclaimer='No standalone four-point v4 decision. No retraining, new ground truth, relabelled training files or changed benchmark metrics. All-real reference includes original full-history information: offline consistency audit only.',
                    versions=dict(numpy=np.__version__, pandas=pd.__version__))
    (args.output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ['# v4生成数据复判：固定真实背景的条件审计', '',
              '## 结论边界', '',
              '本审计按v4阈值重新判断生成Measure，而不是直接沿用标签。但四点样本不能独立执行完整v4；下述结果是在真实来源32点窗口及真实同组设备背景中的条件判定。正常=未触发v4候选规则，不等于确认健康；异常=候选，不等于确认故障。', '',
              '## 方法', '',
              '1. 每个样本单独放回真实来源的四个位置，其他真实数据保持不变，不拼接不同样本。',
              '2. 固定原始环境预测、MAD、分层参考分布和窗口尺度，重新计算四点残差/同组差值及跳变（含接回真实序列的边界跳变）。不把生成的证据通道当成可信检测输入。',
              '3. 使用32点窗口和原v4各项阈值；跨32点边界的四点样本取触及窗口任一候选为正类。仅针对来源设备输出判定，真实peer背景固定。',
              '4. 原v4分层rank没有out-of-sample接口：本审计采用平均秩概率的单调线性延拓，极端值clip。它是固定参考的审计扩展，不是每个样本重新拟合全套v4。',
              '5. 对照将四点替换为原环境预测值。若对照仍异常，说明真实背景或与peer差异足以触发，不足以证明生成四点保留异常。internal_peer_trigger只检查四点内部跳变，不使用外部接缝，仍不等于完整v4独立复判。',
              '6. 缺peer数据不伪造为零；保留原v4无peer分支。非正Measure另列物理审计，不自动等同v4异常。', '',
              f'校验：原始{len(checker.windows)}个v4窗口全部标签一致；{checker.real_roundtrip_count}个真实四点来源窗口回代全部一致。', '',
              '## 三种子合并结果（统计标签一致性，不是检测准确率）', '',
              '|比例|方法|正常→正常|正常→候选|正类→正常|正类→候选|正常一致率|正类一致率|正类的基线替换对照仍候选|正类四点内部跳变触发|',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in pooled.itertuples():
        report.append(f'|{r.ratio}|{r.method}|{r.normal_to_normal}|{r.normal_to_candidate}|{r.positive_to_normal}|{r.positive_to_candidate}|{r.normal_agreement:.2%}|{r.positive_agreement:.2%}|{r.positive_neutral_context_candidates}|{r.positive_internal_peer_triggers}|')
    report += ['', '## 文件', '', '- `rechecked_*.csv.gz`：逐样本新判定、触发原因、原标签、来源、内部跳变及物理检查。',
               '- `configuration_summary.csv`：每个比例/方法/seed汇总与输入SHA256。', '- `method_summary.csv`：三种子合并计数。',
               '- `device_summary.csv`：逐设备汇总。', '- `protocol.json`：固定背景/标定、阈值、秩延拓和局限。', '',
               '这些结果不能直接替代原数据增强标签或重算原分类成绩；如需新训练，需要另设无信息泄漏的标定与验证流程。', '']
    (args.output / 'V4_GENERATED_RECHECK.md').write_text('\n'.join(report), encoding='utf-8')
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=ROOT / '07_generated_v4_recheck')
    parser.add_argument('--ratios', nargs='+')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    main(parser.parse_args())
