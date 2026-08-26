import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-04-09'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=P.pct_change(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].rename('VIX').loc[:end]
v=vix.shift(1); reg=(v>v.rolling(60,min_periods=40).quantile(.75)).astype(float)
r10=P.shift(1)/P.shift(11)-1; vol20=R.shift(1).rolling(20,min_periods=15).std()
sig=(-r10/(np.sqrt(10)*vol20)).mul(reg,axis=0)
rows=[]
for dt in P.index:
 x=sig.loc[dt]; y=P.shift(-10).loc[dt]/P.loc[dt]-1
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.x.nunique()>1: rows.append((dt,len(z),spearmanr(z.x,z.y).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); mean=r.ic.mean(); sd=r.ic.std(ddof=1)
print('dates',len(r),'start',r.index.min(),'end',r.index.max(),'avg_n',r.n.mean())
print('IC',mean,'ICIR_daily',mean/sd,'hit',(r.ic>0).mean(),'coverage',r.n.mean()/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-04-09')]:
 q=r.loc[a:b]; print(a,b,'dates',len(q),'IC',q.ic.mean() if len(q) else np.nan,'ICIR',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
sig.to_csv('scripts/miner_3_20290409_vix_regime_reversal_signal.csv')
