from pathlib import Path
import argparse
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "ratio_benchmark_20260918" / "final_test_train"


def md(df):
    lines=["| " + " | ".join(df.columns) + " |", "|"+"|".join(["---"]*len(df.columns))+"|"]
    for _,r in df.iterrows():
        vals=[]
        for c in df.columns:
            v=r[c]
            vals.append(f"{v:.4f}" if isinstance(v,float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main(run=RUN):
    global RUN
    RUN = Path(run)
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    stable_filter = bool(protocol.get('stable_stage_filter', False))
    result=pd.read_csv(RUN/'all_results.csv')
    # Keep one baseline row per ratio/classifier; the runner repeats it once per method.
    base=result[result.method=='Real-only'].drop_duplicates(['ratio','classifier'])
    aug=result[result.method!='Real-only']
    best=[]
    for ratio,g in aug.groupby('ratio'):
        g=g.copy(); g['score']=g['ROC-AUC']+g['Average Precision']
        best.append(g.sort_values('score',ascending=False).iloc[0])
    best=pd.DataFrame(best)
    best_table=best[['ratio','method','classifier','F1','ROC-AUC','Average Precision','Precision','Recall']].round(4)
    base_table=base[['ratio','classifier','F1','ROC-AUC','Average Precision','Precision','Recall']].round(4)
    delta_rows=[]
    for _, row in best.iterrows():
        b=base[(base.ratio == row.ratio) & (base.classifier == row.classifier)].iloc[0]
        delta_rows.append({
            'ratio': row.ratio,
            'method': row.method,
            'classifier': row.classifier,
            'F1_delta': row['F1'] - b['F1'],
            'ROC-AUC_delta': row['ROC-AUC'] - b['ROC-AUC'],
            'AP_delta': row['Average Precision'] - b['Average Precision'],
        })
    delta_table=pd.DataFrame(delta_rows).round(4)
    method_summary=aug.groupby(['ratio','method','classifier']).agg(F1=('F1','mean'),ROC_AUC=('ROC-AUC','mean'),AP=('Average Precision','mean'),Precision=('Precision','mean'),Recall=('Recall','mean')).reset_index()
    method_summary['score']=method_summary.ROC_AUC+method_summary.AP
    metric_winners=[]
    for ratio, g in aug.groupby('ratio'):
        for metric in ('F1', 'ROC-AUC', 'Average Precision'):
            row=g.loc[g[metric].idxmax()]
            metric_winners.append({'ratio': ratio, 'metric': metric, 'method': row.method, 'classifier': row.classifier, 'value': row[metric]})
    metric_winners=pd.DataFrame(metric_winners).round(4)
    physical_path=RUN/'raw_generation_physics_audit.csv'
    physical_best=None
    if physical_path.exists():
        physical=pd.read_csv(physical_path)
        physical_best=(physical.sort_values(['ratio','window_any_nonpositive_rate','outside_source_date_any_value_rate','relation_diff_mae'])
                       .groupby('ratio',as_index=False).first()[['ratio','method','window_any_nonpositive_rate','outside_source_date_any_value_rate','relation_diff_mae','relation_ratio_mae','generated_m4_hum_corr','source_real_m4_hum_corr']]
                       .round(4))
    out=RUN/'ratio_all_papers_summary.xlsx'
    with pd.ExcelWriter(out,engine='openpyxl') as xl:
        result.to_excel(xl,sheet_name='all_results',index=False)
        method_summary.sort_values(['ratio','score'],ascending=[True,False]).to_excel(xl,sheet_name='method_summary',index=False)
        best_table.to_excel(xl,sheet_name='best_by_ratio',index=False)
        base_table.to_excel(xl,sheet_name='real_baseline',index=False)
        delta_table.to_excel(xl,sheet_name='best_vs_baseline',index=False)
        metric_winners.to_excel(xl,sheet_name='metric_winners',index=False)
        if physical_best is not None:
            physical_best.to_excel(xl,sheet_name='raw_physics_best',index=False)
    lines=['# Updated experiment.xlsx: all paper schemes under train:test ratios','',
           '## Protocol','',
           '- Input: updated `experiment.xlsx`, including dates 250605, 250626, 250718, and 260910.',
           '- Pairing: nearest timestamp within 40 seconds; only equal error labels retained.',
           '- Features: `M4_residual`, `M7_residual`, `M4-M7`, `M4/M7`, `log(M4/M7)`; window length 4.',
            '- Split: chronological within each `(date, error stage)` group using the test:train names `2:8`, `3:7`, `4:6`, `5:5`. The training portion is split 80/20 into fit/validation and the tail is held out test.',
           '- Methods: 5 fixed upstream adapters plus 13 paper-mechanism adapters (18 total).',
           '- Seed: 42; epochs: 15; diffusion steps: 20; one synthetic sample per real training window per class.',
           '- Metrics are evaluated on real held-out windows. Raw generation is used; no physical clipping is applied.', '',
           '## Real-only baseline','',md(base_table),'',
            '## Best paper scheme per ratio','',md(best_table),'',
            '## Best scheme versus real-only baseline','',md(delta_table),'',
            '## Winners by individual metric','',md(metric_winners),'',
            '## Lowest raw physical violation per ratio','',
            md(physical_best) if physical_best is not None else 'Raw audit not available.', '',
            '## Interpretation','',
            '- The best method changes with the train:test ratio; there is no universally dominant paper scheme.',
            '- Increasing the training fraction generally changes the test distribution and threshold selection, so metrics should be compared within the same ratio.',
            '- These results are a first updated-data sweep with one seed and reduced local training budget; they are not yet a 3-seed publication-grade estimate.',
            ('- Stable-stage filter is applied: normal data uses the terminal stable suffix and abnormal phases discard the first two matched points.' if stable_filter else '- No stable-stage-only filter is applied in this run; all matched error phases and their within-phase windows are eligible.'),
            ('- The stable-only run has much fewer windows (and small held-out sets), so near-perfect scores are not directly comparable to the all-phase run and require multi-seed confirmation.' if stable_filter else ''),
            '- The new 260910 date is included in the same chronological protocol and changes the distribution compared with the previous three-date experiments.', '']
    (RUN/'RATIO_ALL_PAPERS_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('wrote',out); print('wrote',RUN/'RATIO_ALL_PAPERS_RESULTS.md'); print('\nBest:'); print(best_table.to_string(index=False))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    main(parser.parse_args().run)
