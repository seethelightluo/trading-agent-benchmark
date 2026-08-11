import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-08-11')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Volatility-normalized medium-term trend with a drawdown-recovery gate.
# The gate rewards positive 20d trend only when price is above its 60d low,
# avoiding raw momentum exposure to persistent drawdowns. Every input is lagged.
ret20=px.pct_change(20); vol20=r.rolling(20,min_periods=15).std(); low60=px.rolling(60,min_periods=40).min(); high60=px.rolling(60,min_periods=40).max(); recovery=(px-low60)/(high60-low60+1e-8)
f=((ret20/(vol20*np.sqrt(20)+1e-8))*(0.5+recovery)).shift(1)
print('universe',len(U),'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];Ns=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): I.append(z);Ns.append(len(q));ds.append(px.index[i])
 a=np.asarray(I); ds=pd.DatetimeIndex(ds)
 print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for label,mask in [('2026+',ds>=pd.Timestamp('2026-01-01')),('2027',ds>=pd.Timestamp('2027-01-01')),('2027Q2',ds>=pd.Timestamp('2027-04-01'))]:
  b=a[mask]; print(label,round(b.mean(),6),round(b.mean()/b.std(ddof=1),6),int(mask.sum()))
rank=f.rank(axis=1,pct=True); print('turnover',round((rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
f.to_csv('scripts/miner_2_20270812_recovery_gated_voltrend_signal.csv',index_label='date')
