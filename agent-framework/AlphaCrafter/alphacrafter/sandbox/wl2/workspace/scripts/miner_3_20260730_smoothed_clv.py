import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=D['SPX'].index
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); H=pd.DataFrame({s:D[s].high.reindex(dates) for s in U}); L=pd.DataFrame({s:D[s].low.reindex(dates) for s in U})
clv=(2*C-H-L)/(H-L).replace(0,np.nan)
# Smoothed lagged candle-location reversal, test one interpretable smoothing window.
for w in [2,3,5]:
 F=-clv.rolling(w,min_periods=w).mean().shift(1)
 for h in [1,5,10]:
  Y=pd.DataFrame({s:D[s].close.shift(-h).div(D[s].close).sub(1).reindex(dates) for s in U});q=[];ns=[];ds=[]
  for dt in dates:
   z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
   if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
  q=np.asarray(q);print('window',w,'h',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
