import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
R={}; F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None: continue
 d=d.sort_values('date').set_index('date'); r=d.close.astype(float).pct_change()
 R[s]=r; F[s]=-(r.rolling(20,min_periods=15).std()/r.rolling(60,min_periods=45).std()).shift(1)
rets=pd.DataFrame(R); f=pd.DataFrame(F); print('assets',len(R),'dates',len(rets),'range',rets.index.min(),rets.index.max())
ics={h:[] for h in [1,5,10]}; dates={h:[] for h in [1,5,10]}; ranks=[]
for dt in f.index:
 for h in ics:
  y=rets.shift(-h).loc[dt]; z=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(z)>=8: ics[h].append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates[h].append(dt)
 if f.loc[dt].notna().sum()>=8:ranks.append(f.loc[dt].rank(pct=True))
for h in ics:
 a=np.array(ics[h]); print(h,'dates',len(a),'avgN',np.mean([len(pd.concat([f.loc[d],rets.shift(-h).loc[d]],axis=1).dropna()) for d in dates[h]]),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().sum(axis=1).mean()/len(R),'turnover',pd.DataFrame(ranks).diff().abs().mean().mean())
for yr in sorted(set(x.year for x in dates[1])):
 a=np.array([v for v,d in zip(ics[1],dates[1]) if d.year==yr]); print(yr,len(a),round(np.nanmean(a),5),round(np.nanmean(a)/np.nanstd(a,ddof=1),4))
print('independent proxy corr with 5d return',f.corrwith(rets.rolling(5).sum()).mean())
