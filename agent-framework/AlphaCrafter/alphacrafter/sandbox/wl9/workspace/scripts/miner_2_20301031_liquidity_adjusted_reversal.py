import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  q=d[['date','close','volume']].copy(); q.date=pd.to_datetime(q.date); data[s]=q.drop_duplicates('date').set_index('date').sort_index()
p=pd.DataFrame({s:x.close.astype(float) for s,x in data.items()}).sort_index()
v=pd.DataFrame({s:x.volume.astype(float) for s,x in data.items()}).reindex(p.index)
r=p.pct_change()
# Liquidity-adjusted short shock: relative 10d underperformance, scaled by downside risk,
# and weighted by abnormal volume so capitulation/reversal events are distinguished.
ret10=p.pct_change(10); rel=ret10.sub(ret10.median(axis=1),axis=0)
down=r.clip(upper=0).rolling(30).std()*np.sqrt(252)
volratio=v.rolling(20).mean().div(v.rolling(60).mean()).replace([np.inf,-np.inf],np.nan)
f=(-rel/down*volratio.clip(0.5,2.0)).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 a=pd.Series(vals,index=dates).dropna(); ic=a.mean(); ir=ic/a.std(ddof=1)*np.sqrt(len(a)) if len(a)>1 else np.nan
 print(f'H={h} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/len(U):.4f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f}')
fr=p.shift(-20).div(p)-1; vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.Series(dict(vals)).sort_index()
for lo,hi,nm in [('2024-01-01','2026-12-31','2024-26'),('2027-01-01','2029-12-31','2027-29'),('2030-01-01','2030-10-30','2030YTD')]:
 q=a.loc[lo:hi]; print(f'regime={nm} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan:.6f}')
print(f'turnover_proxy={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} instruments={len(data)} dates={len(p)} avg_factor_N={f.notna().sum(axis=1).mean():.2f}')
f.index=f.index.strftime('%Y-%m-%d'); f.to_csv('scripts/miner_2_20301031_liquidity_adjusted_reversal_signal.csv',index_label='date')
