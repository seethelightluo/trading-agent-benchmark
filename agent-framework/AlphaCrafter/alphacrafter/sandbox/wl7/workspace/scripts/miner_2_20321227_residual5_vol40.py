import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close
px=pd.DataFrame(D).sort_index(); r=px.pct_change(); r5=px.pct_change(5); resid=r5.sub(r5.median(axis=1),axis=0)
vol=r.rolling(40,min_periods=20).std(); f=-resid/(vol+1e-8)
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);print('H',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/(np.std(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',np.nanmean(f.rank(axis=1,pct=True).diff().abs().mean(axis=1)))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_2_20321227_residual5_vol40_signal.csv',index=False)
