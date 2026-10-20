"""Report Figure 4 (worked example, 28 Apr 2026). Stored results only, nothing refitted. Output folder: $REPORT_FIG_DIR (default reports/figures/report/).

Run from the repo root: python scripts/report/fig4_worked_example.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
R = str(ROOT / "reports" / "forecast") + "/"
F = os.environ.get("REPORT_FIG_DIR", str(ROOT / "reports" / "figures" / "report")) + "/"
os.makedirs(F, exist_ok=True)
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
QC,CC='#E0A030','#2E6FA8'
s,e='2026-04-28 09:50','2026-04-28 11:00'
d=pd.read_parquet(R+'final_cps_test_rows.parquet').loc[s:e]
q=pd.read_parquet(R+'qra_test_forecasts.parquet').loc[s:e]
assert (abs(d.y-q.y)<1e-9).all()
fig,ax=plt.subplots(figsize=(9.5,4.6))
import numpy as np
off=pd.Timedelta('50s'); bw=pd.Timedelta('90s')
ax.bar(d.index-off,d.cps_split_hi-d.cps_split_lo,bottom=d.cps_split_lo,width=bw,color=CC,alpha=0.55,label='CPS 90% range')
ax.bar(q.index+off,q['q0.950']-q['q0.050'],bottom=q['q0.050'],width=bw,color=QC,alpha=0.55,label='QRA 90% range')
ax.plot(d.index,d.regime,'_',color=CC,ms=12,mew=2,label='Best guess (regime switch, CPS centre)')
ax.plot(q.index,q['q0.500'],'_',color='#9a6400',ms=12,mew=2,label='QRA middle (0.5 level)')
ax.plot(d.index,d.y,'k.-',lw=1,label='Actual price')
ax.axhline(170.318,ls=':',color='r',lw=1,label='Top of normal band (170.3)')
ax.set_ylabel('$/MWh'); ax.set_title('Worked example: 28 April 2026, 09:50 to 11:00 (interval end times)'); ax.legend(loc='upper left',fontsize=8.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.tight_layout(); fig.savefig(F+'v3_worked_example_cps_qra.png',dpi=200)
for t in ['10:25','10:40','10:45']:
    ts=pd.Timestamp('2026-04-28 '+t); print(t,round(q.loc[ts,'q0.050'],2),round(q.loc[ts,'q0.950'],2),round(d.loc[ts,'cps_split_lo'],2),round(d.loc[ts,'cps_split_hi'],2))
