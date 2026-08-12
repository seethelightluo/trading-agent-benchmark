import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-09-05')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Relative volatility-adjusted reversal: recent 20d return, scaled by 20d realized vol; lag one day.
vol=r.rolling(20,min_periods=10).std(); raw=-(px/px.shift(20)-1)/(vol*np.sqrt(20)+1e-12); f=raw.sub(raw.mean(axis=1),axis=0).shift(1)
print('vol_adjusted_reversal',len(px),px.index.max().date(),'assets',len(U))
for h in [5,10,20]:
 I=[];D=[]; Ns=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   I.append(spearmanr(q.f,q.y).statistic);D.append(px.index[i]);Ns.append(len(q))
 a=np.array(I); d=pd.DatetimeIndex(D)
 print('H',h,'dates',len(a),'avgN',round(np.mean(Ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
 for name,mask in [('2028+',d>=pd.Timestamp('2028-01-01')),('2029YTD',d>=pd.Timestamp('2029-01-01')),('2029Aug+',d>=pd.Timestamp('2029-08-01'))]:
  x=a[mask]
  print(name,len(x),round(x.mean(),6) if len(x) else None,round(x.mean()/(x.std(ddof=1)+1e-12),6) if len(x)>1 else None)
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20290906_vol_adjusted_reversal_signal.csv',index=False)
