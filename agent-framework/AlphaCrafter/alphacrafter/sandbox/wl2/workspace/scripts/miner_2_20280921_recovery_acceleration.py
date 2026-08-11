import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
U=[s for s in U if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,4000)
 if d is not None: D[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(D).sort_index(); r=P.pct_change()
# Recovery acceleration: change in risk-adjusted recovery from trailing 60d low over 10 completed bars.
low=P.rolling(60,min_periods=40).min(); vol=r.rolling(20,min_periods=15).std()
eff=(P/low-1)/(vol*np.sqrt(20)); f=(eff-eff.shift(10)).shift(1)
print('universe',len(P.columns),'span',P.index.min(),P.index.max())
rows=[]
for h in [1,3,5,10,20]:
 fr=P.pct_change(h).shift(-h); qs=[]; ns=[]; ds=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=ds).dropna(); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),3))
 if h==5:
  for dt in q.index:
   for s in f.columns: rows.append({'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':f.loc[dt,s]})
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(rows).to_csv('scripts/miner_2_20280921_recovery_acceleration_signal.csv',index=False)
