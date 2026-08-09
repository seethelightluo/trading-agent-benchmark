import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in syms}
close=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); ret=close.pct_change()
# candidate: short-term reversal, volatility scaled, conditional on broad cross-asset dispersion
vol=ret.rolling(20).std(); dispersion=ret.median(axis=1).rolling(5).std()
for name,sig in [('rev1',-ret/(vol+1e-8)),('rev3',-ret.rolling(3).sum()/(vol*np.sqrt(3)+1e-8)),('rev5',-ret.rolling(5).sum()/(vol*np.sqrt(5)+1e-8)),('disp_rev3',-ret.rolling(3).sum()/(vol*np.sqrt(3)+1e-8)* (dispersion>dispersion.rolling(60).median()).astype(float).values[:,None])]:
 sig=pd.DataFrame(sig,index=close.index,columns=syms)
 print('\n',name)
 for h in [1,5,10]:
  fwd=close.pct_change(h).shift(-h)
  arr=[]; dates=[]; ns=[]
  for dt in sig.index:
   a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(a)>=8:
    z=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
    if np.isfinite(z): arr.append(z); dates.append(dt); ns.append(len(a))
  x=pd.Series(arr,index=dates); print(h,len(x),round(np.mean(ns),2),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
  if h==1:
   for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
    q=x[(x.index.year>=int(lo))&(x.index.year<=int(hi))]; print(lo+'-'+hi,round(q.mean(),6),len(q))
 if name=='disp_rev3': sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_1_20270325_dispersion_reversal_signal.csv',index=False)
