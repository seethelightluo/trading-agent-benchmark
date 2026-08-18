import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change(); vol=r.rolling(20).std();
# medium-horizon reversal, scaled by risk, with a persistence penalty for extreme volatility
F=(-P.pct_change(20)/(vol*np.sqrt(20))).shift(1)
for h in [1,5,10,20]:
 a=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a)),6),'hit',round(np.mean(a>0),4))
print('coverage',F.notna().mean().mean(),'turnover',F.rank(pct=True).diff().abs().mean(axis=1).mean(),'dates',len(P),'instruments',len(U))
