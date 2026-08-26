import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; dct={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); dct[s]=x.drop_duplicates('date').set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(dct).sort_index(); r=p.pct_change(); mom=p.pct_change(60); vol=r.rolling(30,min_periods=20).std(); short=p.pct_change(5)
rel=mom.sub(mom.median(axis=1),axis=0); f=(-(rel/(vol+1e-8)-.35*short/(vol+1e-8))).shift(1)
for h in [10,20,40,60]:
 fr=p.shift(-h).div(p)-1; vals=[]; ns=[]; ix=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ix.append(dt)
 a=pd.Series(vals,index=ix).dropna(); print(f'H={h} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(252):.6f} hit={np.mean(a>0):.4f}')
 if h==60:
  for lo,hi,n in [('2024','2026','2024-26'),('2027','2029','2027-29'),('2030','2030','2030YTD')]:
   q=a[(a.index>=lo)&(a.index<=hi)]; print(f' regime={n} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.6f}')
f.to_csv('scripts/miner_1_20300905_relative_momentum_reversal_signal.csv',index_label='date'); print(f'turnover_proxy={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} instruments={len(dct)} dates={len(p)}')