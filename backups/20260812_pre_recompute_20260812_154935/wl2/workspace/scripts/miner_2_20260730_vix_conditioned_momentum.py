import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index; P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=P.pct_change()
I=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(dates).ffill()
# VIX-conditioned momentum: 20d cross-sectional residual momentum, reversed in high-volatility regime.
mom=P.pct_change(20); resid=mom.sub(mom.median(axis=1),axis=0)
reg=(I.pct_change(5)>0).astype(float)*2-1
F=(resid.mul(reg,axis=0)).shift(1)
for h in [1,5,10]:
 Y=P.shift(-h).div(P).sub(1); q=[]; ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v):q.append(v);ns.append(len(z))
 q=np.array(q); print('horizon',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  print('regimes',[(int(v),round(np.mean(q[np.array([reg.loc[d] for d in dates if d in F.index and len(pd.DataFrame({"f":F.loc[d],"y":Y.loc[d]}).dropna())>=8])==v]),6) if np.any(np.array([reg.loc[d] for d in dates if d in F.index and len(pd.DataFrame({"f":F.loc[d],"y":Y.loc[d]}).dropna())>=8])==v) else None) for v in [-1,1]])
rank=F.rank(axis=1,pct=True); print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
