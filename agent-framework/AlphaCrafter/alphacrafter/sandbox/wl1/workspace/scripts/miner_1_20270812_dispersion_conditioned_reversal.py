import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-08-12')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}; ix=sorted(set().union(*[set(x.index) for x in P.values()])); p=pd.DataFrame({s:x.reindex(ix) for s,x in P.items() for _ in [0]}).ffill(); r=p.pct_change()
# Dispersion-conditioned short reversal: negate 5d return only when cross-asset
# dispersion is elevated; otherwise use a weak 20d trend anchor.
disp=r.std(axis=1).rolling(20,min_periods=15).rank(pct=True); short=r.rolling(5,min_periods=4).sum(); trend=r.rolling(20,min_periods=15).sum(); f=np.where(disp.values[:,None]>.65,-short,0.25*trend); f=pd.DataFrame(f,index=p.index,columns=p.columns).shift(1)
for h in [5,10,20]:
 I=[];N=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: I.append(spearmanr(q.f,q.y).statistic);N.append(len(q))
 a=np.array(I);print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover',round(f.rank(pct=True).diff().abs().stack().groupby(level=0).mean().dropna().mean(),6))
