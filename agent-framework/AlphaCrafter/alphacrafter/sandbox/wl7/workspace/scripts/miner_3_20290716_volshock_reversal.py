import numpy as np,pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2029-07-16'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]; R=P.pct_change(); lag=R.shift(1)
# Volatility-shock reversal: reverse prior 5d return, risk scaled, only when cross-asset dispersion is above trailing median.
disp=lag.std(axis=1).where(lag.notna().sum(axis=1)>=8); med=disp.rolling(252,min_periods=126).median().shift(1); gate=(disp>=med).astype(float)
sig=-(P.shift(1)/P.shift(6)-1)/R.shift(1).rolling(20,min_periods=15).std().mul(gate,axis=0).replace(0,np.nan)
for h in [5,10,20]:
 f=P.shift(-h)/P-1; out=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.x.nunique()>1: out.append((dt,len(z),spearmanr(z.x,z.y).statistic))
 r=pd.DataFrame(out,columns=['date','n','ic']).set_index('date').dropna(); mu=r.ic.mean(); ir=mu/r.ic.std(ddof=1)
 print('horizon',h,'dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.mean()/15,4),'IC',round(mu,6),'ICIR',round(ir,6),'hit',round((r.ic>0).mean(),4))
 for a,b in [('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-07-16')]:
  q=r.loc[a:b].ic; print('regime',a,b,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
 if h==10:r.to_csv('scripts/miner_3_20290716_volshock_reversal_ic.csv')
sig.to_csv('scripts/miner_3_20290716_volshock_reversal_signal.csv'); print('active_days',round(gate.mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
