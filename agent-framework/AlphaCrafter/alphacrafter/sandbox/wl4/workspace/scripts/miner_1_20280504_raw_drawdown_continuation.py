import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),usecols=['date','close']);d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close
p=pd.DataFrame(P).sort_index(); fr=p.shift(-1)/p-1
for w in [60,90,120]:
 high=p.rolling(w,min_periods=int(w*.65)).max().shift(1); f=(-(p.shift(1)/high-1)).replace([np.inf,-np.inf],np.nan); f=f.sub(f.median(axis=1),axis=0)
 z=[];ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[dt][ok],fr.loc[dt][ok]).statistic
   if np.isfinite(q):z.append(q);ns.append(ok.sum())
 x=pd.Series(z); print('window=%d dates=%d avg_n=%.2f coverage=%.4f IC=%.6f ICIR=%.6f hit=%.4f recent250=%.6f'%(w,len(x),np.mean(ns),np.mean(ns)/15,x.mean(),x.mean()/x.std(),np.mean(x>0),x.tail(250).mean()))
