import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; fs={}
for s in U:
 p=f'{base}/{s}.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); gap=d.open/d.close.shift(1)-1
  d['signal']=-gap/gap.rolling(20,min_periods=10).std(); d['fwd']=d.close.shift(-1)/d.close-1; fs[s]=d[['date','signal','fwd']]
D=sorted(set.intersection(*[set(x.date) for x in fs.values()])); rows=[]; sr=[]
for dt in D:
 v=[]; y=[]
 for s,d in fs.items():
  q=d[d.date==dt]
  if len(q) and np.isfinite(q.signal.iloc[0]) and np.isfinite(q.fwd.iloc[0]): v.append(q.signal.iloc[0]); y.append(q.fwd.iloc[0]); sr.append({'date':dt,'symbol':s,'signal':q.signal.iloc[0]})
 if len(v)>=8:
  ic=spearmanr(v,y).statistic
  if np.isfinite(ic): rows.append((dt,ic,len(v)))
r=pd.DataFrame(rows,columns=['date','ic','n']); print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/len(r)/15,'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
 x=r[(r.date.dt.year>=a)&(r.date.dt.year<=b)]; print(a,b,len(x),x.ic.mean() if len(x) else np.nan,x.ic.mean()/x.ic.std(ddof=1) if len(x)>1 else np.nan)
z=pd.DataFrame(sr); p=z.pivot(index='date',columns='symbol',values='signal').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean().mean()); out='../persistent/factor_signals_miner_1_20270226_volscaled_gap_reversal.csv'; z.to_csv(out,index=False); print('artifact',out)
