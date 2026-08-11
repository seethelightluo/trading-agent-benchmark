import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2027-04-08')
def L(s):return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
D={s:L(s).loc[:cut] for s in U};ix=sorted(set().union(*[set(x.index) for x in D.values()]));p=pd.DataFrame({s:x.close.reindex(ix) for s,x in D.items()}).ffill();r=p.pct_change();
# Breadth-confirmed medium momentum: return times excess positive-day breadth, rewarding persistent trends rather than one-off jumps.
mom=p.pct_change(20); breadth=(r>0).rolling(20,min_periods=15).mean(); vol=r.rolling(40,min_periods=25).std(); f=(mom/vol*(0.5+breadth)).shift(1)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[];N=[];ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);N.append(len(q));ds.append(p.index[i])
 a=np.array(I);print('h',h,'valid_dates',len(a),'avgN',round(np.mean(N),2),'coverage',round(np.mean(N)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==10:print('annual10',{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True);print('turnover',round((rank-rank.shift()).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
