import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:'2030-10-16']; R=P.pct_change()
def e(w): return P.pct_change(w)/(R.abs().rolling(w,min_periods=max(8,w//2)).sum()+1e-12)
F=-(e(10)-e(40))
def calc(h):
 z=[]
 for i in range(41,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:z.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(z,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h);m=x.ic.mean();print(h,len(x),x.n.mean(),x.n.mean()/15,m,m/(x.ic.std(ddof=1))*np.sqrt(len(x)),(x.ic>0).mean())
x=calc(20);print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string());print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean(),'end',P.index.max().date())
