import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; r=d.close.pct_change();
 # compression-conditioned directional breakout: lagged 5d trend amplified by low short/long vol
 short=r.rolling(5,min_periods=4).std(); long=r.rolling(30,min_periods=15).std(); comp=(1-short/(long+1e-12)).clip(-2,2)
 d['factor']=d.close.pct_change(5).shift(1)*comp.shift(1)
 for h in [1,5,10]: d[f'y{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows,ignore_index=True)
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1 and g[f'y{h}'].nunique()>1:
   q=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(q):a.append((dt,q,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']); q=z.ic
 print('H',h,'dates',len(q),'avgN',round(z.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.date.dt.year):print('YR',yr,'IC',round(g.ic.mean(),5),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),4))
print('coverage',round(x.factor.notna().mean(),4),'turnover',round(x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
x.to_csv('scripts/miner_3_20261217_compression_breakout_signal.csv',index=False)
