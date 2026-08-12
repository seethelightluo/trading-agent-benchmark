import pandas as pd, numpy as np, glob
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-07-23')
xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]
 xs[s]=d
p=pd.DataFrame(xs).sort_index().ffill()
# one interpretable factor: rebound from trailing low, conditioned on trend and downside risk
ret20=p.pct_change(20); ret60=p.pct_change(60)
dd=p/p.rolling(60).max()-1
vol=p.pct_change().rolling(40).std()*np.sqrt(252)
# positive recovery / risk, with trend confirmation; ranks reduce scale effects
f=(ret20 - 0.5*dd.abs())/(vol+1e-8) + 0.35*ret60/(vol+1e-8)
# lag is inherent through end t, evaluate forward after t
rows=[]
for h in [1,5,10,20]:
  ic=[]
  for dt in p.index:
    a=f.loc[dt]; y=p.shift(-h).loc[dt]/p.loc[dt]-1
    z=pd.concat([a,y],axis=1).dropna()
    if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
  q=pd.Series(ic).dropna(); rows.append((h,len(q),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean()))
print('data dates',len(p),'assets',p.notna().sum().mean(),'end',p.index.max())
print('horizon dates IC ICIR hit')
for x in rows: print(x)
# regimes 20d
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
 ic=[]
 for dt in p.index:
  if not (str(a)<=str(dt.year)<=str(b)): continue
  z=pd.concat([f.loc[dt],(p.shift(-20).loc[dt]/p.loc[dt]-1)],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(ic).dropna(); print('regime',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# coverage and turnover based rank signal
valid=f.notna().sum(axis=1)/15
rank=f.rank(axis=1,pct=True)
to=rank.diff().abs().mean(axis=1).mean()
print('coverage',valid.mean(),'turnover_proxy',to)
# artifact for reproducibility
out=pd.DataFrame({s:f[s] for s in U}); out.index.name='date'; out.to_csv('scripts/miner_2_20310724_recovery_risk_signal.csv')
