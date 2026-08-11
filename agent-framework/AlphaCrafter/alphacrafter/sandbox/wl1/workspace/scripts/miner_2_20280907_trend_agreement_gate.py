import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2028-09-06')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}; idx=sorted(set().union(*map(lambda x:set(x.index),P.values())));px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill();r=px.pct_change();v=r.rolling(20,min_periods=15).std()
a=r.rolling(60,min_periods=40).sum()/(v*np.sqrt(60)+1e-8); b=r.rolling(20,min_periods=15).sum()/(v*np.sqrt(20)+1e-8)
# 60d trend strength, rewarded when short and medium trends agree, otherwise damped
f=(a*(0.5+0.5*np.sign(a)*np.sign(b))).shift(1)
print('factor trend_agreement_gate_60d universe',15,'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(px.index[i])
 x=np.asarray(I);d=pd.DatetimeIndex(ds);print('h',h,'valid_dates',len(x),'avgN',np.mean(N),'coverage',np.mean(N)/15,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0))
 for lab,m in [('2026+',d>=pd.Timestamp('2026-01-01')),('2027+',d>=pd.Timestamp('2027-01-01')),('2028+',d>=pd.Timestamp('2028-01-01'))]:
  z=x[m];print(lab,z.mean(),z.mean()/z.std(ddof=1),len(z))
rank=f.rank(axis=1,pct=True);print('turnover',((rank-rank.shift()).abs().stack().groupby(level=0).mean().dropna().mean()))
f.to_csv('scripts/miner_2_20280907_trend_agreement_gate_signal.csv',index_label='date')
