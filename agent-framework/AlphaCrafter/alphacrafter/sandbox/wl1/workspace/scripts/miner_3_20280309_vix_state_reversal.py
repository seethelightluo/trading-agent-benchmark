import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2028-03-08')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut]
idx=sorted(set().union(*[set(x.index) for x in P.values()]).union(vix.index)); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); v=vix.reindex(idx).ffill(); r=px.pct_change()
vix_z=(v-v.rolling(252,min_periods=100).mean())/(v.rolling(252,min_periods=100).std()+1e-9)
stress=(1+0.5*vix_z.clip(-1,2)).clip(0.25,2.0)
f=(-r.rolling(5,min_periods=5).sum().mul(stress,axis=0)).shift(1)
print('factor vix_state_reversal_5d universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 a=np.array(I); ds=pd.DatetimeIndex(ds)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,m in [('2025+',ds>=pd.Timestamp('2025-01-01')),('2026+',ds>=pd.Timestamp('2026-01-01')),('2027+',ds>=pd.Timestamp('2027-01-01')),('2028+',ds>=pd.Timestamp('2028-01-01'))]:
  z=a[m]; print(lab,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'dates',int(m.sum()))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
f.to_csv('scripts/miner_3_20280309_vix_state_reversal_signal.csv',index_label='date')
