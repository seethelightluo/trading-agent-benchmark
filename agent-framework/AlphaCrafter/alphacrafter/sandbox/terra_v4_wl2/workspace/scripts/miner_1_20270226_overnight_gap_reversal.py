import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
frames={}
for s in U:
    p=os.path.join(base,s+'.csv')
    if os.path.exists(p):
        d=pd.read_csv(p,parse_dates=['date']).sort_values('date')
        d['prev_close']=d.close.shift(1)
        # prior completed day's overnight gap; larger positive gap expected to reverse
        d['signal']=-(d.open/d.prev_close-1.0)
        d['fwd']=d.close.shift(-1)/d.close-1.0
        frames[s]=d[['date','signal','fwd']]
all_dates=sorted(set.intersection(*[set(x.date) for x in frames.values()]))
rows=[]; sigrows=[]
for dt in all_dates:
    vals=[]; fwd=[]
    for s,d in frames.items():
        q=d[d.date==dt]
        if len(q) and np.isfinite(q.signal.iloc[0]) and np.isfinite(q.fwd.iloc[0]):
            vals.append(q.signal.iloc[0]); fwd.append(q.fwd.iloc[0])
            sigrows.append({'date':dt,'symbol':s,'signal':q.signal.iloc[0]})
    if len(vals)>=8:
        ic=spearmanr(vals,fwd).statistic
        if np.isfinite(ic): rows.append((dt,ic,len(vals)))
r=pd.DataFrame(rows,columns=['date','ic','n']);
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15),'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit', (r.ic>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 x=r[(r.date.dt.year>=int(a))&(r.date.dt.year<=int(b))]; print(a,b,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1) if len(x)>1 else np.nan)
# rank turnover
z=pd.DataFrame(sigrows); z['rank']=z.groupby('date').signal.rank(pct=True); piv=z.pivot(index='date',columns='symbol',values='rank').sort_index(); print('turnover',piv.diff().abs().mean().mean())
os.makedirs('../persistent',exist_ok=True); out='../persistent/factor_signals_miner_1_20270226_overnight_gap_reversal.csv'; z.to_csv(out,index=False); print('artifact',out)
