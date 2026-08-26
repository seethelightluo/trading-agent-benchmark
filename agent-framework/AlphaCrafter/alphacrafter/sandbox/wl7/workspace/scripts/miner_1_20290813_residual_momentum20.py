import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-08-13'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=P.pct_change(); lag=R.shift(1)
# Cross-sectional residual momentum: prior 20d asset return less contemporaneous universe median,
# scaled by prior 40d idiosyncratic volatility. All inputs end before decision date.
r20=P.shift(1)/P.shift(21)-1
mkt=r20.median(axis=1).where(r20.notna().sum(axis=1)>=8)
res=r20.sub(mkt,axis=0)
idvol=lag.sub(lag.median(axis=1),axis=0).rolling(40,min_periods=25).std()
sig=(res/idvol).replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [5,10,20]:
 f=P.shift(-h)/P-1; out=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1: out.append((dt,len(z),spearmanr(z.x,z.y).statistic))
 r=pd.DataFrame(out,columns=['date','n','ic']).set_index('date').dropna(); m=r.ic.mean(); sd=r.ic.std(ddof=1)
 print('horizon',h,'dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.mean()/15,4),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((r.ic>0).mean(),4))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-08-13')]:
  q=r.loc[a:b].ic; print('regime',a,b,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
 if h==20: r.to_csv('scripts/miner_1_20290813_residual_momentum20_ic.csv')
sig.to_csv('scripts/miner_1_20290813_residual_momentum20_signal.csv')
print('dates',len(P),'assets',P.notna().sum().to_dict())
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
