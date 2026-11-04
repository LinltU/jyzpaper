"""Read-only, print-sized figures for the concise v4 progress report."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_v4_generated_data import (
    load_context, load_generated, add_windows, set_limits, merge_intervals,
    BLUE, RED, GRAY, PINK,
)
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
DEVICES = ['863644079729449', '863644079730603', '863644079730645', '863644079730694']
METHOD = 'IMDiffusion'


def text_font(fig):
    from matplotlib.text import Text
    for item in fig.findobj(Text):
        prop = FONT.copy()
        prop.set_size(item.get_fontsize())
        item.set_fontproperties(prop)


def save(fig, output):
    text_font(fig)
    fig.savefig(output, dpi=260, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overview(out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), layout='constrained')
    axes[0].bar(['正常', '不同步跳变', '孤立极端跳变', '残差与同组一致'], [509, 31, 4, 3],
                color=[GRAY, RED, '#f97316', '#eab308'])
    axes[0].set_title('547个窗口的标注组成')
    axes[0].set_ylabel('窗口数')
    for i, n in enumerate([509, 31, 4, 3]):
        axes[0].text(i, n + 9, str(n), ha='center', fontsize=10)
    axes[0].set_ylim(0, 560)
    axes[1].bar(['全体匹配点基准', '原始同组差值', 'v4分层标准分'], [70.3, 97.5, 70.9],
                color=[GRAY, '#8b5cf6', RED])
    axes[1].axhline(70.3, color=GRAY, linestyle='--', linewidth=1)
    axes[1].set_title('偏离最大的5%点：高湿占比')
    axes[1].set_ylabel('湿度>60的点占比（%）')
    axes[1].set_ylim(0, 112)
    for i, n in enumerate([70.3, 97.5, 70.9]):
        axes[1].text(i, n + 2, f'{n:.1f}%', ha='center', fontsize=10)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    save(fig, out / 'anomaly_audit.png')

    best = pd.read_csv(ROOT / '03_benchmark_v4_evidence_features/formal_v4_best_methods.csv')
    fig, ax = plt.subplots(figsize=(10.5, 4.8), layout='constrained')
    x = np.arange(len(best))
    ax.bar(x - .18, best.real_only_ap, .34, label='仅真实数据', color=BLUE)
    ax.bar(x + .18, best['Average Precision_mean'], .34,
           yerr=best['Average Precision_std'], capsize=4, label='最佳增强（三种子均值±标准差）', color='#f97316')
    ax.set_xticks(x, [f"{r.ratio.replace('to', ':')}\n{r.method}" for r in best.itertuples()])
    ax.set_ylabel('Average Precision（AP）')
    ax.set_xlabel('训练:测试比例 / 汇总最佳方法')
    ax.set_ylim(0, .34)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    for i, row in best.iterrows():
        ax.text(i - .18, row.real_only_ap + .006, f'{row.real_only_ap:.4f}', ha='center', fontsize=10)
        ax.text(i + .18, row['Average Precision_mean'] + row['Average Precision_std'] + .007,
                f"{row['Average Precision_mean']:.4f}", ha='center', fontsize=10)
    save(fig, out / 'best_vs_real.png')


def generated(out):
    run = ROOT / '03_benchmark_v4_evidence_features'
    best = pd.read_csv(run / 'formal_v4_best_methods.csv')
    selected = best.loc[best['Average Precision_mean'].idxmax()]
    assert selected['ratio'] == '4to6' and selected['method'] == METHOD
    ctx = load_context(run)
    values, labels, sources = load_generated(run / f'raw_4to6_{METHOD}_seed42.npz', ctx, '4to6')
    assert np.allclose(values[:, :2], ctx.real[sources, :2, 0], rtol=1e-5, atol=1e-6), 'Expected retained first two observations'
    device_ids = ctx.meta.IMEI.to_numpy()[sources]
    times = dates.date2num(ctx.times[sources])
    manifest = []
    for kind in ['overlay', 'local', 'localization']:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.0))
        for ax, imei in zip(axes.ravel(), DEVICES):
            ids = np.flatnonzero(device_ids == imei)
            positives = ids[labels[ids] == 1]
            p = ctx.points[imei]
            rt = dates.date2num(p.Timestamp.to_numpy())
            if kind == 'local':
                pool = positives if len(positives) else ids
                pool = pool[np.argsort(sources[pool], kind='stable')]
                j = int(pool[len(pool) // 2])
                sid = int(sources[j])
                ax.plot(times[j], ctx.real[sid, :, 0], 'o-', color=GRAY, linewidth=1.4, markersize=4)
                ax.plot(times[j], values[j], 'o-', color=RED if labels[j] else BLUE, linewidth=1.4, markersize=4)
                set_limits(ax, times[j], np.r_[ctx.real[sid, :, 0], values[j]])
                title = f'{imei}\n来源={sid}，类别={labels[j]}；前两点保留，后两点生成'
                manifest.append({'IMEI': imei, 'sample': j, 'source': sid, 'label': int(labels[j])})
            else:
                ax.plot(rt, p.Measure, color=GRAY, linewidth=.7, alpha=.7)
                for row in ctx.candidates.get(imei, pd.DataFrame()).itertuples():
                    ax.axvspan(dates.date2num(row.start_time), dates.date2num(row.end_time),
                               color=PINK, alpha=.35, linewidth=0)
                displayed = ids if kind == 'overlay' else positives
                for label, color in [(0, BLUE), (1, RED)]:
                    sel = displayed[labels[displayed] == label]
                    if len(sel):
                        add_windows(ax, times[sel], values[sel], color)
                set_limits(ax, rt, np.r_[p.Measure.to_numpy(), values[displayed].ravel()], bottom_room=kind == 'localization')
                if kind == 'localization':
                    for label, color, y in [(0, BLUE, .025), (1, RED, .075)]:
                        sel = ids[labels[ids] == label]
                        intervals = merge_intervals([(times[j, 0], times[j, -1]) for j in sel])
                        if intervals:
                            rug = LineCollection([[(a, y), (b, y)] for a, b in intervals], colors=color,
                                                 linewidths=3, transform=ax.get_xaxis_transform())
                            ax.add_collection(rug, autolim=False)
                    if not len(positives):
                        ax.text(.5, .85, '无正类生成窗口', transform=ax.transAxes, ha='center', fontsize=9)
                title = f'{imei}\n生成窗口={len(ids)}，其中正类={len(positives)}'
            ax.set_title(title, fontsize=9)
            locator = dates.AutoDateLocator(minticks=2, maxticks=8, interval_multiples=False)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
            ax.tick_params(axis='x', labelsize=8, rotation=15)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylabel('Measure', fontsize=9)
            ax.grid(alpha=.18)
        if kind == 'local':
            handles = [Line2D([], [], color=GRAY, marker='o', label='真实来源四点'),
                       Line2D([], [], color=BLUE, marker='o', label='生成正常类'),
                       Line2D([], [], color=RED, marker='o', label='生成正类（继承标签）')]
        else:
            handles = [Line2D([], [], color=GRAY, label='真实Measure'),
                       Line2D([], [], color=BLUE, label='生成正常类' if kind == 'overlay' else '底部蓝条：正常来源覆盖'),
                       Line2D([], [], color=RED, label='生成正类（继承标签）'),
                       Patch(facecolor=PINK, label='真实v4候选区间')]
        fig.suptitle(f'{METHOD} / 训练:测试=4:6 / seed=42', fontsize=12, y=.998)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .95), ncol=2, fontsize=9, frameon=False)
        fig.subplots_adjust(top=.83, bottom=.1, left=.075, right=.985, hspace=.55, wspace=.25)
        save(fig, out / f'imdiffusion_{kind}.png')
    (out / 'source_selection.json').write_text(json.dumps({
        'configuration': f'v4-evi / {METHOD} / train:test=4:6 / seed=42',
        'selection_metric': 'Highest three-seed HGB AP mean in the current best-method summary; not a generative-quality ranking.',
        'ap_mean': float(selected['Average Precision_mean']),
        'ap_std': float(selected['Average Precision_std']),
        'devices': DEVICES,
        'selection_note': 'Same four devices as the previous report. No new device or sample selection based on the appearance of IMDiffusion outputs.',
        'all_windows_shown_for_selected_devices': True,
        'local_selection': 'Prefer positive class, source-id-sorted middle sample; same as full grid.',
        'samples': manifest,
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    overview(args.output)
    generated(args.output)
    print('Prepared five figures:', args.output)
