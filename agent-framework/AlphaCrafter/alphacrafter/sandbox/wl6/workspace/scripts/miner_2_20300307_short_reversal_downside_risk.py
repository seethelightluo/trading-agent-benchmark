import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2030-03-07'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); vol=R.rolling(20,min_periods=10).std()*np.sqrt(20)
# Short-horizon reversal, risk-normalized and lagged one completed session.
F=(-(P/P.shift(5)-1)/vol).shift(1)

def evaluate(h):
 out=[]
 for i in range(25,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: out.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 x=pd.DataFrame(out,columns=['date','ic','n']); ic=x.ic.mean(); ir=ic/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
 print('horizon',h,'dates',len(x),'avg_n',round(x.n.mean(),3),'coverage',round(x.n.sum()/(len(x)*15),6),'IC',round(ic,6),'ICIR',round(ir,4),'hit',round((x.ic>0).mean(),4))
 print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
for h in [5,10,20]: evaluate(h)
# turnover proxy: average cross-sectional rank movement on weekly observations
q=F.rank(axis=1,pct=True); print('turnover_proxy',float(q.diff().abs().mean().mean()))
