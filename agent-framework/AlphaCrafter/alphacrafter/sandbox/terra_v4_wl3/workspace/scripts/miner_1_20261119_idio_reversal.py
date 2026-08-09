import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); P[s]=d.close
P=pd.DataFrame(P).sort_index().loc[:'2026-11-18']; R=P.pct_change(fill_method=None)
# Idiosyncratic short reversal: remove equal-weight cross-asset daily return, then
# rank assets by negative 3-day residual return, scaled by trailing 20d residual volatility.
M=R.mean(axis=1); resid=R.sub(M,axis=0)
rv=resid.rolling(20,min_periods=15).std()
F=(-resid.rolling(3,min_periods=3).sum()/rv.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_1_20261119_idio_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h,fill_method=None).shift(-h); a=[]; ns=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
 if h==1:
  for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
   q=a[[str(P.index[j].year)>=lo and str(P.index[j].year)<=hi for j in range(len(P)-h) if len(pd.concat([F.iloc[j],Y.iloc[j]],axis=1).dropna())>=8]]
   print('REG',lo,hi,len(q),round(np.nanmean(q),6))
print('coverage',round(F.notna().sum().sum()/(len(F)*15),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',P.index.min(),P.index.max(),'assets',P.shape[1])
