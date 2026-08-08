import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for a in A}).sort_index(); R=P.pct_change(fill_method=None)
# One idea: slower volatility-normalized mean reversion aligned to 10-day rebalance.
# Signal is lagged to ensure only completed observations are used.
rev=-(P/P.shift(10)-1)
vol=R.rolling(30,min_periods=20).std()
F=(rev/(vol*np.sqrt(10)+1e-12)).shift(1)
F=F.sub(F.mean(axis=1),axis=0)
print('idea=10d volatility-normalized reversal rows',len(P),'assets',len(A),'valid_cells',int(F.notna().sum().sum()),'coverage',round(float(F.notna().mean().mean()),6))
def ev(h):
 fw=P.shift(-h)/P-1; vals=[]; ds=[]; ns=[]
 for t in F.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   vals.append(spearmanr(q.f,q.r).statistic);ds.append(t);ns.append(len(q))
 x=np.asarray(vals); return pd.Series(x,index=ds),np.mean(ns)
for h in [1,5,10,20]:
 x,n=ev(h);print('H',h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(float((x>0).mean()),4),'meanN',round(n,2))
x,n=ev(10)
for lo,hi in [('2020','2025-12-31'),('2026','2030-12-31'),('2031','2035-12-05')]:
 y=x.loc[lo:hi]; print('regime',lo,hi,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6))
print('turnover_daily_rank',round(float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))
print('LIBRARY_AUDIT=FAILED exact aligned histories for all admitted factors not reconstructed; no persistence')
