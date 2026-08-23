import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:'2030-10-02'];R=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill(); vr=v.rolling(60,min_periods=30).rank(pct=True).to_numpy()[:,None]
r10=P.pct_change(10); dd=P/P.rolling(60,min_periods=30).max()-1
F=(r10-0.5*dd).mul(1-vr[:,0],axis=0)+(-dd+0.25*r10).mul(vr[:,0],axis=0)
def calc(h):
 o=[]
 for i in range(65,len(P)-h):
  a=F.iloc[i].values;b=(P.iloc[i+h]/P.iloc[i]-1).values;ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:o.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
 return pd.DataFrame(o,columns=['date','ic','n'])
for h in [5,10,20]:
 x=calc(h);m=x.ic.mean();ir=m/x.ic.std(ddof=1)*np.sqrt(len(x));print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
x=calc(10);print('turnover_proxy',F.rank(axis=1,pct=True).diff().abs().mean().mean());print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string());print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())