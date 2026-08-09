import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 p='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); v=d.volume.replace(0,np.nan)
 shock=(v.shift(1)/(v.shift(2).rolling(20,min_periods=10).median()+1e-12)-1).clip(lower=0)
 d['factor']=d.close.shift(1).pct_change(20)*np.log1p(shock)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True); results={}
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1 and g[f'y{h}'].nunique()>1:
   z=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(z): a.append((dt,z,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']); q=z.ic
 print('H',h,'dates',len(q),'avgN',round(z.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.date.dt.year): print('YR',yr,'dates',len(g),'IC',round(g.ic.mean(),5),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),4))
 results[h]=z
v=x.dropna(subset=['factor']); print('coverage',round(len(v)/len(x),4),'turnover',round(v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
# artifact is the exact factor signal and forward returns for audit
x.to_csv('scripts/miner_3_20261217_volume_trend_continuation_signal.csv',index=False)
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
