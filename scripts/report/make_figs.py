"""Report Figures 2, 3 and 5. Stored results only, nothing refitted. Output folder: $REPORT_FIG_DIR (default reports/figures/report/).

Run from the repo root: python scripts/report/make_figs.py
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
R = str(ROOT / "reports" / "forecast") + "/"
F = os.environ.get("REPORT_FIG_DIR", str(ROOT / "reports" / "figures" / "report")) + "/"
os.makedirs(F, exist_ok=True)
"""Figures for v3, built only from stored result files (no refitting)."""
import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
cq=pd.read_csv(R+'cps_qra_comparison.csv').set_index(['system','subset'])
bt=pd.read_csv(R+'cps_qra_bootstrap.csv').set_index(['metric','subset'])
regs=['all','calm','tail','onset']; names=['All intervals\n(40,127)','Calm\n(36,921)','Tail\n(3,206)','Spike starts\n(403)']
QC,CC='#E0A030','#2E6FA8'
def ci(metric,r,fmt):
    b=bt.loc[(metric,r)]; return ('CPS−QRA '+fmt+'\n['+fmt+', '+fmt+']').format(b.diff_cps_minus_qra,b.boot_p05,b.boot_p95)
# Fig 2: coverage and width by subset
fig,ax=plt.subplots(1,2,figsize=(11,4.4)); x=np.arange(4); w=0.36
for k,(col,lab,yl,dy,f) in enumerate((('coverage_90','How often the 90% range held the price','90% coverage on test',0.01,'{:.3f}'),('width_90','How wide the 90% range was','Mean width of 90% range ($/MWh)',0.5,'{:.1f}'))):
    cv=[cq.loc[('cps',r),col] for r in regs]; qv=[cq.loc[('qra',r),col] for r in regs]
    ax[k].bar(x-w/2,cv,w,color=CC,label='CPS'); ax[k].bar(x+w/2,qv,w,color=QC,label='QRA')
    for i in range(4):
        ax[k].text(x[i]-w/2,cv[i]+dy,f.format(cv[i]),ha='center',fontsize=8.5); ax[k].text(x[i]+w/2,qv[i]+dy,f.format(qv[i]),ha='center',fontsize=8.5)
    ax[k].set_xticks(x,names,fontsize=9); ax[k].set_ylabel(yl); ax[k].set_title(lab)
ax[0].axhline(0.9,ls='--',color='k',lw=1); ax[0].text(3.6,0.885,'target 0.90',fontsize=8,ha='right',va='top'); ax[0].set_ylim(0,1.08); ax[0].legend(loc='upper right',bbox_to_anchor=(1,0.8))
ax[1].set_ylim(0,40)
fig.suptitle('CPS and QRA 90% ranges on the test period (number of intervals in brackets)'); fig.tight_layout(); fig.savefig(F+'v3_cps_qra_coverage_width.png',dpi=200); plt.close(fig)
# Fig 3: CRPS and interval score, with paired bootstrap CIs
fig,axs=plt.subplots(2,4,figsize=(12,6.6))
for row,(col,metric,yl,fmt) in enumerate((('crps','crps (QRA full)','CRPS ($/MWh)','{:+.3f}'),('interval_score_90','interval_score_90','90% interval score ($/MWh)','{:+.1f}'))):
    for j,r in enumerate(regs):
        a=axs[row,j]; cv=cq.loc[('cps',r),col]; qv=cq.loc[('qra',r),col]
        a.bar([0,1],[cv,qv],0.6,color=[CC,QC],edgecolor='k')
        for xx,v in ((0,cv),(1,qv)): a.text(xx,v*1.02,(f'{v:.3f}' if v<10 else f'{v:.2f}') if row==0 else f'{v:.1f}',ha='center',fontsize=8.5)
        a.set_xticks([0,1],['CPS','QRA']); a.set_ylim(0,max(cv,qv)*1.45)
        a.text(0.5,0.97,ci(metric,r,('{:+.4f}' if r=='all' else fmt) if row==0 and r in ('all','calm') else (fmt if row==1 else '{:+.2f}')),transform=a.transAxes,ha='center',va='top',fontsize=7.5)
        if row==0: a.set_title(names[j].replace('\n',' '),fontsize=10)
    axs[row,0].set_ylabel(yl+'\nlower is better')
fig.text(0.5,0.005,'Both CRPS values are exact scores of a full distribution (QRA: 23 levels joined by straight lines, held flat below 0.01 and above 0.99).\n'
         'Brackets: 90% one-day block-bootstrap confidence interval of the paired difference.',ha='center',fontsize=8.5)
fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig(F+'v3_cps_qra_crps.png',dpi=200); plt.close(fig)
# Fig: variants coverage vs width
m=pd.read_csv(R+'final_master_table.csv'); t=m[m.split=='test'].set_index('model')
pts=[('cps_split','CPS (main)','CPS'),
     ('qra','QRA (main)','QRA'),('qra_variant_selected','QRA, last 28 days','QRA'),('qave_variant','Q-Ave, last 28 days','QRA')]
fig,ax=plt.subplots(figsize=(7.5,4.6))
for k,lab,fam in pts:
    xv,yv=t.loc[k,'width_90'],t.loc[k,'coverage_90']
    ax.scatter(xv,yv,s=90,color=QC if fam=='QRA' else CC,marker='o' if fam=='QRA' else 's',edgecolor='k',zorder=3)
    off={'qra':(8,-12),'cps_split':(-150,6),'qave_variant':(-135,4),'cps_pit':(-175,-14),'cps_mondrian_sliding':(10,-22),'qra_variant_selected':(10,-4)}[k]
    ax.annotate(f'{lab}\n({yv:.3f}, {xv:.1f})',(xv,yv),textcoords='offset points',xytext=off,fontsize=8.5)
ax.axhline(0.9,ls='--',color='k',lw=1); ax.set_xlabel('Mean width of 90% range ($/MWh), test'); ax.set_ylabel('90% coverage, test')
ax.set_xlim(14,33); ax.set_ylim(0.85,0.945); ax.text(32.8,0.901,'target 0.90',fontsize=8,ha='right',va='bottom')
ax.scatter([],[],color=CC,marker='s',edgecolor='k',label='CPS'); ax.scatter([],[],color=QC,marker='o',edgecolor='k',label='QRA family'); ax.legend(loc='lower right')
ax.set_title('Coverage against width: the two main methods and three QRA variants'); fig.tight_layout(); fig.savefig(F+'v3_cps_qra_variants_coverage_width.png',dpi=200); plt.close(fig)
print('ok')
