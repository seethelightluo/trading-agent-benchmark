import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:end]
R=P.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill(); vr=v.pct_change()
# Negative rolling VIX beta: assets with favorable (less negative / positive) VIX sensitivity rank higher.
F=pd.DataFrame({s:-(R[s].rolling(60,min_periods=45).cov(vr)/vr.rolling(60,min_periods=45).var()) for s in U})
for h in [1,5,10]:
  vals=[]; ns=[]; reg={}
  for i in range(60,len(P)-h):
    z=pd.concat([F.iloc[i],(P.iloc[i+h]/P.iloc[i]-1).rename('fwd')],axis=1).dropna()
    if len(z)>=8:
      q=spearmanr(z.iloc[:,0],z.fwd).statistic; vals.append(q);ns.append(len(z));reg.setdefault(str(P.index[i].year),[]).append(q)
  a=np.array(vals); print('horizon',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'regimes',{k:round(np.mean(x),4) for k,x in reg.items()})
print('coverage',F.notna().sum().sum()/(len(F)*15),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean()*2)
# correlations to implementable price factors, pooled rank values
factors={'short_term_reversal_5d':-R.rolling(5).sum(),'peer_median_leadlag_5d':pd.DataFrame({s:R.rolling(5).sum().drop(columns=s).median(axis=1) for s in U}),'risk_adjusted_momentum_20d':R.rolling(20).sum()/R.rolling(20).std()}
for n,x in factors.items(): print('corr',n,F.stack().corr(x.stack()))
