"""Compare position-conditioned outputs of several paper mechanisms."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/position_conditioned_other_papers_20260917'
METHODS=('NsDiff','TimeGrad','FIDE','WaveStitch','TabDDPM')
DATES=('250605','250626','250718')
FEATURES=5

def main():
    meta=pd.read_json(RUN/'metadata.json',dtype={'date':str}).reset_index(drop=True)
    audit=pd.read_csv(RUN/'paired_row_audit.csv',dtype={'date':str})
    rows=[]; summaries=[]
    with pd.ExcelWriter(RUN/'other_papers_error_position_raw_vs_real.xlsx',engine='openpyxl') as xl:
        for method in METHODS:
            for date in DATES:
                p=RUN/'generators'/method/'seed42'/date
                if not (p/'synthetic.npz').exists(): continue
                syn=np.load(p/'synthetic.npz'); sm=pd.read_json(p/'result.json') if False else None
                sd=pd.read_json(p/'metadata.json',dtype={'date':str}) if (p/'metadata.json').exists() else None
                # result metadata is not separate; generated labels/anchors are saved in npz.
                labels=syn['label'].astype(int); anchors=syn['anchor_ids'].astype(int); raw=syn['raw']
                real=audit[audit.date==date].sort_values('dt')
                part=[]
                for r in real.itertuples(index=False):
                    part.append({'source':'real','method':'experiment.xlsx','date':date,'error_position':int(r.error4),'step':0,'Measure4':r.Measure4,'Measure7':r.Measure7,'flag':'real'})
                for i in range(min(len(labels),400)):
                    a=meta.iloc[anchors[i]]
                    for step in range(4):
                        m4=float(raw[i,step,0]+a.base4); m7=float(raw[i,step,1]+a.base7)
                        part.append({'source':'generated_raw','method':method,'date':date,'error_position':int(labels[i]),'step':step,'Measure4':m4,'Measure7':m7,'flag':'raw_unmodified'})
                frame=pd.DataFrame(part)
                frame[['Measure4','Measure7']]=frame[['Measure4','Measure7']].round(6)
                sheet=f'{method[:5]}_{date}'
                frame.to_excel(xl,sheet_name=sheet,index=False)
                ws=xl.sheets[sheet]
                for c in ws[1]: c.font=Font(bold=True); c.alignment=Alignment(horizontal='center')
                fill=PatternFill('solid',fgColor='FCE4D6')
                for rr in range(2,ws.max_row+1):
                    if ws.cell(rr,1).value=='generated_raw':
                        for cc in range(1,ws.max_column+1): ws.cell(rr,cc).fill=fill
                    for cc in range(1,ws.max_column+1): ws.cell(rr,cc).alignment=Alignment(horizontal='center')
                ws.freeze_panes='A2'
                rows.append(frame)
                for error in sorted(set(labels.tolist())):
                    x=raw[labels==error]
                    rg=real[real.error4==error]
                    summaries.append({'method':method,'date':date,'error_position':error,'generated_windows':len(x),'real_rows':len(rg),'M4_mean':float(x[:,:,0].mean()+meta.loc[anchors[labels==error],'base4'].mean()) if len(x) else np.nan,'M7_mean':float(x[:,:,1].mean()+meta.loc[anchors[labels==error],'base7'].mean()) if len(x) else np.nan,'M4_min':float((x[:,:,0]+meta.loc[anchors[labels==error],'base4'].to_numpy()[:,None]).min()),'M4_max':float((x[:,:,0]+meta.loc[anchors[labels==error],'base4'].to_numpy()[:,None]).max()),'M7_min':float((x[:,:,1]+meta.loc[anchors[labels==error],'base7'].to_numpy()[:,None]).min()),'M7_max':float((x[:,:,1]+meta.loc[anchors[labels==error],'base7'].to_numpy()[:,None]).max()),'M4_nonpositive_pct':float(((x[:,:,0]+meta.loc[anchors[labels==error],'base4'].to_numpy()[:,None])<=0).mean()*100),'M7_nonpositive_pct':float(((x[:,:,1]+meta.loc[anchors[labels==error],'base7'].to_numpy()[:,None])<=0).mean()*100)})
        summary=pd.DataFrame(summaries).round(6)
        summary.to_excel(xl,sheet_name='position_audit',index=False)
        pd.DataFrame({'item':['methods','raw policy','error positions','missing positions'],'value':['NsDiff, TimeGrad, FIDE, WaveStitch, TabDDPM','Raw values from each saved synthetic.npz; no projection or filtering','0,1,2,3,5,6,8,9','4 and 7 absent from experiment.xlsx']}).to_excel(xl,sheet_name='notes',index=False)
    out=pd.concat(rows,ignore_index=True)
    out.to_csv(RUN/'other_papers_error_position_raw_vs_real.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(RUN/'other_papers_error_position_audit.csv',index=False,encoding='utf-8-sig')
    print('wrote',RUN/'other_papers_error_position_raw_vs_real.xlsx')
    print(summary.groupby('method')[['M4_nonpositive_pct','M7_nonpositive_pct','M4_min','M4_max','M7_min','M7_max']].mean().round(4).to_string())

if __name__=='__main__': main()
