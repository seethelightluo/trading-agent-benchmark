import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}; dates=D['SPX'].index;R=pd.DataFrame({s:D[s].close.pct_change().reindex(dates) for s in U})
for bw,vw in [(8,20),(12,10),(12,40),(20,20),(20,40),(20,60),(30,40),(30,60)]:
 b=R.gt(0).rolling(bw,min_periods=max(6,bw-3)).mean()-R.lt(0).rolling(bw,min_periods=max(6,bw-3)).mean(); F=(b/R.rolling(vw,min_periods=max(10,vw-8)).std()).shift(1);Y=pd.DataFrame({s:D[s].close.shift(-1).div(D[s].close).sub(1).reindex(dates) for s in U});q=[];ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.array(q);print(bw,vw,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4),round(F.notna().sum().sum()/F.size,4))
