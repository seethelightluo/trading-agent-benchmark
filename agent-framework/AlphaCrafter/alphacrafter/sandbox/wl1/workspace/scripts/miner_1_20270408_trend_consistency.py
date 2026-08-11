import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-04-08')
def L(base,s): return pd.read_csv(base+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close
P={s:L('../persistent/stock_data/',s).loc[:cut] for s in U}; ix=sorted(set().union(*map(set,[x.index for x in P.values()]))); p=pd.DataFrame({s:x.reindex(ix) for s,x in P.items()}).ffill(); r=p.pct_change()
# Trend consistency: medium horizon return, scaled by volatility, and gated by fraction of positive sessions.
ret=p.pct_change(60); vol=r.rolling(60,min_periods=40).std(); consistency=(r>0).rolling(60,min_periods=40).mean(); f=(ret/vol*(0.5+consistency)).shift(1)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];N=[];D=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));D.append(p.index[i])
 a=np.array(I);print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==20:print('annual20',{y:round(a[[d.year==y for d in D]].mean(),6) for y in sorted(set(d.year for d in D))})
rank=f.rank(axis=1,pct=True);print('turnover',round((rank-rank.shift()).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
