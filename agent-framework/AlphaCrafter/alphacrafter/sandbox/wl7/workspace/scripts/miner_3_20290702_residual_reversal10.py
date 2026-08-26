import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); end=pd.Timestamp('2029-07-02')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=P.pct_change(); m=R.mean(axis=1)
# Residualized short reversal: negative lagged 10d return after removing rolling 60d beta to equal-weight cross-asset movement, risk scaled.
lag=R.shift(1); cov=lag.rolling(60,min_periods=40).cov(m.shift(0)); vm=m.rolling(60,min_periods=40).var()
beta=cov.div(vm,axis=0); resid10=-(P.shift(1)/P.shift(11)-1 - beta.mul(m.rolling(10,min_periods=8).sum(),axis=0))
vol=R.shift(1).rolling(20,min_periods=15).std(); sig=resid10/vol
rows=[]
for h in [5,10,20]:
 f=P.shift(-h)/P-1; out=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.x.nunique()>1: out.append((dt,len(z),spearmanr(z.x,z.y).statistic))
 r=pd.DataFrame(out,columns=['date','n','ic']).set_index('date').dropna(); mean=r.ic.mean(); sd=r.ic.std(ddof=1)
 print('horizon',h,'dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.mean()/15,4),'IC',round(mean,6),'ICIR',round(mean/sd,6),'hit',round((r.ic>0).mean(),4))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-07-02')]:
  z=r.loc[a:b].ic; print('regime',a,b,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
 if h==10: r.to_csv('scripts/miner_3_20290702_residual_reversal10_ic.csv')
sig.to_csv('scripts/miner_3_20290702_residual_reversal10_signal.csv')
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
