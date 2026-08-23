import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2030-11-13'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
V=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['volume'].rename(s) for s in U],axis=1).sort_index().reindex(P.index)
R=P.pct_change(); lv=np.log(V.replace(0,np.nan)); vs=(lv.rolling(20,min_periods=10).mean()-lv.rolling(60,min_periods=30).mean())/(lv.rolling(60,min_periods=30).std()+1e-12)
F=R.rolling(10).sum()*(1+0.25*vs.clip(-3,3))
def calc(h):
 out=[]; turns=[]; ranks=F.rank(axis=1,pct=True)
 for i in range(80,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:
   out.append((P.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum())); turns.append(np.mean(np.abs(ranks.iloc[i]-ranks.iloc[i-1])))
 return pd.DataFrame(out,columns=['date','ic','n']),turns
for h in [5,10,20]:
 x,t=calc(h); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)); print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
x,t=calc(10); print('turnover_proxy',np.mean(t)); print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string()); print('data_dates',len(P),'instruments',len(U),'data_end',P.index.max().date())
