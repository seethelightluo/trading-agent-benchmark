import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close
px=pd.DataFrame(D).sort_index(); ret=px.pct_change()
# volatility-scaled 1-day reversal, lagged naturally at date t predicting t+H
f=-ret/(ret.rolling(20,min_periods=10).std()+1e-8)
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; v=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 v=np.array(v); print(h,len(v),round(np.mean(ns),2),f'{np.mean(v):.6f}',f'{np.mean(v)/(np.std(v,ddof=1)+1e-12):.6f}',f'{np.mean(v>0):.4f}')
rr=f.rank(axis=1,pct=True); print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',np.nanmean(rr.diff().abs().mean(axis=1)))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('scripts/miner_2_20321227_1d_vol_reversal_signal.csv',index=False)
