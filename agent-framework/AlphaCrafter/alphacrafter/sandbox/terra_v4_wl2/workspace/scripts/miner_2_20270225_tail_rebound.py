import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
# Tail-rebound: buy assets with a sharp but not ongoing 20d drawdown, using only completed close data.
# Signal is 3d reversal activated when 20d drawdown is below -8%; forward return is h-day close-to-close.
rets={a:p[a].pct_change() for a in A}; sig={}
for a in A:
    draw=p[a]/p[a].rolling(20,min_periods=20).max()-1
    sig[a]=(-p[a].pct_change(3)).where(draw < -0.08)

def run(h):
    fwd={a:p[a].shift(-h)/p[a]-1 for a in A}; ics=[]; ns=[]; rows=[]
    dates=sorted(set().union(*[sig[a].index for a in A]))
    for d in dates:
        x=[]; y=[]
        for a in A:
            if d in sig[a].index and d in fwd[a].index:
                x.append(sig[a].loc[d]); y.append(fwd[a].loc[d])
        z=pd.DataFrame({'x':x,'y':y}).dropna()
        if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
            ic=spearmanr(z.x,z.y).statistic
            if np.isfinite(ic): ics.append(ic); ns.append(len(z)); rows.append((d,ic))
    q=np.array(ics); print('H',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(len(q)/len(dates),4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
    for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
        v=[ic for d,ic in rows if lo<=str(d)[:7 if '-' in lo else 4]<=hi]
        if v: print(lo, len(v), round(np.mean(v),6), round(np.mean(v)/np.std(v,ddof=1),6))
    return q,rows
for h in [1,5,10]: run(h)
# rank turnover on consecutive valid dates
print('signal_dates',sum(sig[a].notna().sum() for a in A))
print('max_abs_library_correlation',None)
