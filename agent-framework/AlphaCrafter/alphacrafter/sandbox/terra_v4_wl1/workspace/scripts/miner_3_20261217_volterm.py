import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); r=d.close.pct_change()
 # low short/long volatility indicates stable regime; negate ratio so high score is calmer and potentially persistent
 vs=r.rolling(5,min_periods=5).std(); vl=r.rolling(30,min_periods=20).std()
 d['factor']=-(vs.shift(1)/(vl.shift(1)+1e-8)-1)
 for h in [1,5,10]: d['y'+str(h)]=d.close.shift(-h)/d.close-1
 rows.append(d[['date','factor','y1','y5','y10']].assign(symbol=s))
x=pd.concat(rows); x.to_csv('scripts/miner_3_20261217_volterm_signal.csv',index=False)
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor','y'+str(h)])
  if len(g)>=8 and g.factor.nunique()>1:
   z=spearmanr(g.factor,g['y'+str(h)]).statistic
   if np.isfinite(z):a.append((dt,z,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']); q=z.ic
 print('H',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for label,mask in [('2020-22',z.date.dt.year<=2022),('2023-24',z.date.dt.year.isin([2023,2024])),('2025-26',z.date.dt.year>=2025)]:
   a=z[mask].ic; print(label,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
print('coverage',round(x.factor.notna().mean(),4),'turnover',round(x.pivot(columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
