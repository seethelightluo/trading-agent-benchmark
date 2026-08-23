import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2030-05-02'); base='../persistent/stock_data'
ps=[]; vs=[]
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')
 ps.append(d['close'].rename(s)); vs.append(d['volume'].rename(s))
P=pd.concat(ps,axis=1).sort_index().loc[:cutoff]; V=pd.concat(vs,axis=1).sort_index().loc[:cutoff]
R=P.pct_change(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
mom=P/P.shift(20)-1; vr=V.rolling(20,min_periods=15).mean()/V.rolling(60,min_periods=40).mean(); F=(mom/(vol+1e-12))*np.sqrt(vr.clip(lower=0.05,upper=20))
for h in [5,10,20]:
 rows=[]
 for i in range(80,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8: rows.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 x=pd.DataFrame(rows,columns=['date','ic','n']);m=x.ic.mean();ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x))
 print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.sum()/(len(x)*15):.5f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
 if h==10: print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data_dates',len(P),'instruments',len(U))
