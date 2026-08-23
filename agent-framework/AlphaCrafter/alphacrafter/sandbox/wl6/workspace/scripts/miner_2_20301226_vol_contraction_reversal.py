import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-12-25'); B='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{B}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cut]
R=P.pct_change(); v10=R.rolling(10,min_periods=8).std(); v40=R.rolling(40,min_periods=30).std()
# Reversal after volatility contraction: inverse of medium momentum scaled by long/short volatility.
F=-P.pct_change(20)*(v40/(v10+1e-12))
def run(h):
 out=[]
 for i in range(80,len(P)-h):
  a=F.iloc[i].values; b=(P.iloc[i+h]/P.iloc[i]-1).values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:
   c=spearmanr(a[ok],b[ok]).statistic
   if np.isfinite(c): out.append((P.index[i],c,ok.sum()))
 x=pd.DataFrame(out,columns=['date','ic','n']); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x)); return x,m,ir
print('data_dates',len(P),'instruments',len(U),'range',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 x,m,ir=run(h); print(f'horizon {h} valid_dates {len(x)} avg_n {x.n.mean():.3f} coverage {x.n.mean()/15:.6f} IC {m:.8f} ICIR {ir:.5f} hit {(x.ic>0).mean():.5f}')
x,m,ir=run(10); print('turnover_proxy',F.rank(axis=1,pct=True).diff().abs().mean().mean()); print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
