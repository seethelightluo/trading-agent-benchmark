import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in syms}
close=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); r=close.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(close.index).ffill()
# medium-term trend, volatility scaled, with VIX level used as defensive conditioning
mom=r.rolling(20).sum(); vol=r.rolling(20).std(); base=mom/(vol+1e-8)
for label, sig in [('plain',base),('highvol',base*(vix>vix.rolling(60).median()).astype(float).values[:,None]),('lowvol',base*(vix<=vix.rolling(60).median()).astype(float).values[:,None])]:
 sig=pd.DataFrame(sig,index=close.index,columns=syms); print('\n',label)
 for h in [1,5,10]:
  fwd=close.pct_change(h).shift(-h); xs=[]; ns=[]
  for dt in sig.index:
   a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(a)>=8:
    z=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
    if np.isfinite(z):xs.append(z);ns.append(len(a))
  x=np.array(xs); print('h',h,'dates',len(x),'N',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 if label=='plain':
  sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_1_20270325_momentum20_signal.csv',index=False)
