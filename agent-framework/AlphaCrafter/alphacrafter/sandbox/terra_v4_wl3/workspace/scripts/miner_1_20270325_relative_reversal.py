import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in syms}
close=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=close.pct_change(); vol=r.rolling(20,min_periods=10).std()
# Relative reversal: fade each asset's recent return relative to contemporaneous cross-asset median, risk scaled.
for k in [1,3,5]:
 sig=-(r.rolling(k).sum()-r.rolling(k).sum().median(axis=1).values[:,None])/(vol*np.sqrt(k)+1e-8)
 sig=pd.DataFrame(sig,index=close.index,columns=syms)
 fwd=close.pct_change().shift(-1); vals=[]; ds=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('k',k,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f turnover %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),sig.notna().sum(axis=1).mean()/15,sig.rank(axis=1,pct=True).diff().abs().mean().mean()))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=s[(s.index.year>=lo)&(s.index.year<=hi)]; print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
 if k==3: sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_1_20270325_relative_reversal_signal.csv',index=False)
