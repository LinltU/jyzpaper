"""Artifact audit plus stratified full-original-v4 refitting sensitivity check."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import warnings
import numpy as np
import pandas as pd
import process_device_nonoverlap_windows_v4 as v4
from audit_v4_generated_recheck import DEFAULT_RUN, ROOT, Rechecker, summary


def original_windows(points):
    # Unmodified original v4 uses nanmax even when a peer-containing window has
    # no valid complete-panel jump. Its NaN result is intended; do not alter it.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        return v4.make_windows(points, 32)


def main(args):
    output = args.output
    table = pd.read_csv(output / 'configuration_summary.csv')
    total = 0
    for row in table.itertuples():
        safe = row.method.replace('-', '_').replace('/', '_')
        stem = f'{row.ratio}_{safe}_seed{row.seed}'
        frame = pd.read_csv(output / f'rechecked_{stem}.csv.gz', dtype={'IMEI': str})
        npz = DEFAULT_RUN / f'raw_{stem}.npz'
        assert hashlib.sha256(npz.read_bytes()).hexdigest() == row.input_sha256, f'Input changed: {npz}'
        with np.load(npz, allow_pickle=False) as z:
            np.testing.assert_array_equal(frame.inherited_label, z['labels'])
            np.testing.assert_array_equal(frame.source_id, z['source_ids'])
            np.testing.assert_array_equal(frame.nonpositive_measure_points, (z['windows'][:, :, 0] <= 0).sum(axis=1))
        assert np.array_equal(frame.sample_id.to_numpy(), np.arange(len(frame)))
        assert frame.rechecked_label.isin([0, 1]).all()
        assert frame.standalone_v4_status.str.startswith('not_evaluable').all()
        stats = summary(frame)
        for name, value in stats.items():
            if isinstance(value, (int, np.integer)):
                assert value == getattr(row, name), (stem, name)
            else:
                # Original point-channel differences are float32; CSV reload is
                # float64 and its mean can differ by several float32 ulps.
                np.testing.assert_allclose(value, getattr(row, name), equal_nan=True, rtol=1e-6, atol=1e-8)
        total += len(frame)
    assert len(table) == 216 and total == 1021518
    print(f'Artifacts and original-input hashes: {len(table)} files / {total} windows PASS', flush=True)
    c = Rechecker(DEFAULT_RUN)
    raw = pd.read_csv(DEFAULT_RUN / 'annotated_device_points_v4.csv', dtype={'IMEI': str})
    raw['Timestamp'] = pd.to_datetime(raw.Timestamp)
    original = raw[['IMEI', 'Timestamp', 'Measure', 'Temp', 'Hum', 'Location', 'group_id']].copy()
    # Actual untouched original-v4 pipeline, without frozen statistics.
    baseline_points = v4.add_point_features(original)
    baseline_windows = original_windows(baseline_points)
    expected = c.windows.set_index('window_id').window_anomaly_v4
    np.testing.assert_array_equal(baseline_windows.set_index('window_id').window_anomaly_v4.reindex(expected.index), expected)
    print('Original v4 full pipeline baseline: 547 windows / 38 candidates PASS', flush=True)
    frame = pd.read_csv(output / 'rechecked_4to6_IMDiffusion_seed42.csv.gz', dtype={'IMEI': str})
    with np.load(DEFAULT_RUN / 'raw_4to6_IMDiffusion_seed42.npz', allow_pickle=False) as z:
        values = z['windows'][:, :, 0]
    chosen = []
    # Evenly spread fixed IDs within each of the four label outcomes; deliberately
    # stratified diagnostic subset, NOT a random representative accuracy sample.
    for _, sub in frame.groupby(['inherited_label', 'rechecked_label'], sort=True):
        chosen.extend(sub.sample_id.to_numpy()[np.linspace(0, len(sub)-1, min(8, len(sub)), dtype=int)].tolist())
    for _, sub in frame[frame.nonpositive_measure_points > 0].groupby('inherited_label'):
        chosen.extend(sub.sample_id.to_numpy()[np.linspace(0, len(sub)-1, min(4, len(sub)), dtype=int)].tolist())
    crosses = [int(row.sample_id) for row in frame.itertuples()
               if c.ctx.meta.iloc[int(row.source_id)].source_rows[0] // 32 != c.ctx.meta.iloc[int(row.source_id)].source_rows[-1] // 32]
    if crosses:
        chosen.extend(np.asarray(crosses)[np.linspace(0, len(crosses)-1, min(4, len(crosses)), dtype=int)].tolist())
    rows = []
    for sample in sorted(set(chosen)):
        info = frame.iloc[sample]
        sid, imei = int(info.source_id), info.IMEI
        locations = np.asarray(c.ctx.meta.iloc[sid].source_rows, int)
        injected = original.copy()
        target = injected[injected.IMEI == imei].sort_values('Timestamp').index.to_numpy()
        injected.loc[target[locations], 'Measure'] = values[sample]
        recalculated = v4.add_point_features(injected)
        windows = original_windows(recalculated)
        affected = windows[(windows.IMEI == imei) & (windows.start_row <= locations[-1]) & (windows.end_row >= locations[0])]
        prediction = int(affected.window_anomaly_v4.any())
        rows.append(dict(sample_id=sample, source_id=sid, IMEI=imei, inherited_label=int(info.inherited_label),
                         frozen_context_label=int(info.rechecked_label), full_refitted_v4_label=prediction,
                         labels_agree=prediction == int(info.rechecked_label),
                         full_refitted_reason=';'.join(sorted(set(affected.loc[affected.window_anomaly_v4==1, 'anomaly_reason']))) or 'normal',
                         nonpositive_measure_points=int(info.nonpositive_measure_points)))
        print(f'Full original v4 sample={sample}: frozen={int(info.rechecked_label)} refitted={prediction}', flush=True)
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(output / 'full_v4_refit_sensitivity.csv', index=False)
    validation = dict(artifact_files=216, generated_windows=total, source_hashes_unchanged=True,
                      real_fourpoint_roundtrips=c.real_roundtrip_count, original_full_pipeline_window_labels_pass=True,
                      sensitivity_samples=len(rows), sensitivity_matching_labels=int(sensitivity.labels_agree.sum()),
                      sensitivity_selection='stratified label outcomes + nonpositive values + crossing 32-point boundaries; NOT representative',
                      limitation='full original-v4 refitting changes peer statistics and global ranks; sensitivity on subset only, not full 1M-window rerun')
    (output / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    report = output / 'V4_GENERATED_RECHECK.md'
    text = report.read_text(encoding='utf-8')
    text = text.split('\n## 完整v4重跑敏感性验证')[0]
    text += f'\n## 完整v4重跑敏感性验证\n\n未修改数据先重跑原始v4：547个窗口标签全部复现。另对IMDiffusion/4:6/seed42的{len(rows)}个分层选定样本逐一注入后，完整重新拟合原v4环境、rank、peer及窗口标定，{int(sensitivity.labels_agree.sum())}个与固定参考判定一致。此子集覆盖标签四种转移、负值与跨32点边界，非随机代表样本，不能外推为整体一致率。详见`full_v4_refit_sensitivity.csv`。\n'
    report.write_text(text, encoding='utf-8')
    # Focused, human-readable outcome report. Conditional agreement is not
    # standalone accuracy, and positivity is a separate physical screen.
    focused = ['# IMDiffusion生成后v4复判摘要（4:6）', '',
               '三种子全部样本已做固定真实背景复判；seed42是当前进展PDF的展示配置。', '',
               '## seed42：5,408个四点窗口', '',
               '|继承类别|复判正常|复判候选异常|类别一致率|', '|---|---:|---:|---:|']
    for label, title in [(0, '正常类'), (1, '正类')]:
        sub = frame[frame.inherited_label == label]
        normal = int((sub.rechecked_label == 0).sum())
        candidate = int((sub.rechecked_label == 1).sum())
        agreement = normal / len(sub) if label == 0 else candidate / len(sub)
        focused.append(f'|{title}|{normal}|{candidate}|{agreement:.2%}|')
    focused += ['', '## 背景和物理有效性对照', '',
                '复判含来源32点窗口其余真实点和真实同组背景，也含生成片段接回真实序列的跳变；不等于四点独立检测。', '',
                '|继承类别|窗口数|含非正Measure|标签一致且四点Measure均正|换为环境预测值后背景仍候选|四点内部peer跳变达到触发阈值|',
                '|---|---:|---:|---:|---:|---:|']
    for label, title in [(0, '正常类'), (1, '正类')]:
        sub = frame[frame.inherited_label == label]
        physical_bad = int((sub.nonpositive_measure_points > 0).sum())
        screened = int(((sub.nonpositive_measure_points == 0) & (sub.rechecked_label == label)).sum())
        focused.append(f'|{title}|{len(sub)}|{physical_bad}|{screened}|{int(sub.neutral_context_label.sum())}|{int(sub.internal_peer_trigger.sum())}|')
    pooled = pd.read_csv(output / 'method_summary.csv')
    best = pooled[(pooled.ratio == '4to6') & (pooled.method == 'IMDiffusion')].iloc[0]
    focused += ['', '## 三种子合计', '',
                f"共{int(best.windows):,}个窗口：正常→正常{int(best.normal_to_normal):,}，正常→候选{int(best.normal_to_candidate):,}，正类→正常{int(best.positive_to_normal):,}，正类→候选{int(best.positive_to_candidate):,}。正常一致率{best.normal_agreement:.2%}，正类一致率{best.positive_agreement:.2%}。", '',
                '## 如何使用', '',
                '- 可以识别条件复判与继承标签不一致的样本，以及非正Measure样本，作为排查和后续筛选依据。',
                '- 正类复判仍异常不能直接证明异常机制被生成器保留：应结合基线替换对照、内部跳变和证据通道一致性逐样本检查。',
                '- `internal_peer_trigger`不包含外部接缝，但可能包含保留的观测点；两次跳变跨32点边界时也不等于同一个v4窗口的重复跳变规则。',
                '- “一致且均正”仅是初步过滤计数，不是正式合格率；没有改变任何训练标签或原始benchmark。',
                '- 四点本身仍不能独立完整复判。温湿度与同组背景为真实来源条件，而非同时生成的整串时序。',
                f'- 原完整v4抽样重跑：{len(rows)}个分层样本，{int(sensitivity.labels_agree.sum())}个与固定参考判定一致；不能外推为整体准确率。', '',
                '## 明细', '', '`rechecked_4to6_IMDiffusion_seed42.csv.gz`保留每个样本的复判结果、触发原因、来源ID和各项对照。', '']
    (output / 'IMDIFFUSION_4TO6_RECHECK.md').write_text('\n'.join(focused), encoding='utf-8')
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    # Publish read-only copies of scripts, not moved source files.
    for filename in ['audit_v4_generated_recheck.py', 'test_audit_v4_generated_recheck.py', 'validate_v4_generated_recheck.py']:
        src = Path(__file__).with_name(filename)
        dest = ROOT / '05_scripts' / filename
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        assert src.read_bytes() == dest.read_bytes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '07_generated_v4_recheck')
    main(parser.parse_args())
