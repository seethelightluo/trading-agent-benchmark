import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  q=d[['date','close','volume']].copy(); q.date=pd.to_datetime(q.date); q=q.drop_duplicates('date').set_index('date').sort_index(); data[s]=q
p=pd.DataFrame({s:x.close.astype(float) for s,x in data.items()}).sort_index(); r=p.pct_change()
# Trend quality: medium relative momentum discounted by realized volatility and short-term reversal.
mom=p.pct_change(60); vol=r.rolling(30,min_periods=20).std(); short=p.pct_change(5)
csmed=mom.median(axis=1); rel=mom.sub(csmed,axis=0)
f=(rel/(vol+1e-8) - 0.35*short/(vol+1e-8)).shift(1)
res=[]
for h in [10,20,40,60]:
 fr=p.shift(-h).div(p)-1; vals=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 a=pd.Series(vals,index=dates).dropna(); ic=a.mean(); sd=a.std(ddof=1); ir=ic/sd*np.sqrt(252) if sd else np.nan
 print(f'H={h} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/len(U):.4f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f}')
 if h==60:
  for lo,hi,nm in [('2024-01-01','2026-12-31','2024-26'),('2027-01-01','2029-12-31','2027-29'),('2030-01-01','2030-12-31','2030YTD')]:
   q=a[(a.index>=lo)&(a.index<=hi)]; print(f' regime={nm} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan:.6f}')
f.to_csv('scripts/miner_1_20300905_relative_momentum_quality_signal.csv',index_label='date')
print(f'turnover_proxy={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} instruments={len(data)} dates={len(p)}')
