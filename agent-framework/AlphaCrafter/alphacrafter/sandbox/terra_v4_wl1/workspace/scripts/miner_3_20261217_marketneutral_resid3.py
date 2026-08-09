import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:END]; R=P.pct_change();
# Market-neutral short-horizon reversal: remove common cross-asset daily shock from 3d return,
# then scale by lagged idiosyncratic volatility. All inputs available before signal date.
peer=R.rolling(3,min_periods=2).sum().median(axis=1)
r3=P.pct_change(3); resid=r3.sub(peer,axis=0)
idvol=R.rolling(20,min_periods=15).std().shift(1)
f=-(resid.shift(1)/(idvol+1e-12))
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; vals=[]; ns=[]
 for dt in P.index:
  q=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.f,q.y).statistic); ns.append(len(q))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('symbols',P.shape[1],'period',P.index.min().date(),P.index.max().date(),'coverage',round(f.notna().sum().sum()/f.size,4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 a=[]
 for dt in P.index:
  if lo<=dt.year<=hi:
   q=pd.concat([f.loc[dt].rename('f'),P.pct_change(-1).loc[dt].rename('y')],axis=1).dropna()
   if len(q)>=8:a.append(spearmanr(q.f,q.y).statistic)
 a=np.array(a); print('REG',lo,hi,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
out=f.stack().rename('factor').reset_index(); out.columns=['date','symbol','factor']; out.to_csv('scripts/miner_3_20261217_marketneutral_resid3_signal.csv',index=False)
