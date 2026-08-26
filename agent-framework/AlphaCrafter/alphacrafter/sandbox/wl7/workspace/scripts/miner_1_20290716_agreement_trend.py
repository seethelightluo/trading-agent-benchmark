import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); end=pd.Timestamp('2029-07-02')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=P.pct_change()
# Agreement-weighted trend: lagged 20d return, strengthened when lagged 5d and 60d returns agree with it.
r5=P.shift(1)/P.shift(6)-1; r20=P.shift(1)/P.shift(21)-1; r60=P.shift(1)/P.shift(61)-1
agreement=((np.sign(r5)==np.sign(r20)).astype(float)+(np.sign(r60)==np.sign(r20)).astype(float))/2
vol=R.shift(1).rolling(20,min_periods=15).std()*np.sqrt(20)
sig=(r20/vol)*agreement

def calc(h):
 f=P.shift(-h)/P-1; rows=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.x.nunique()>1: rows.append((dt,len(z),spearmanr(z.x,z.y).statistic))
 return pd.DataFrame(rows,columns=['date','n','ic']).set_index('date').dropna()
r=calc(10); m=r.ic.mean(); sd=r.ic.std(ddof=1)
print('assets',len(U),'dates',len(r),'start',r.index.min(),'end',r.index.max(),'avg_n',r.n.mean())
print('IC',m,'ICIR_daily',m/sd,'hit',(r.ic>0).mean(),'coverage',r.n.mean()/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-07-02')]:
 z=r.loc[a:b].ic; print('regime',a,b,'dates',len(z),'IC',z.mean() if len(z) else np.nan,'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [5,10,20]:
 q=calc(h); print('decay',h,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'dates',len(q))
sig.to_csv('scripts/miner_1_20290716_agreement_trend_signal.csv'); r.to_csv('scripts/miner_1_20290716_agreement_trend_ic.csv')
