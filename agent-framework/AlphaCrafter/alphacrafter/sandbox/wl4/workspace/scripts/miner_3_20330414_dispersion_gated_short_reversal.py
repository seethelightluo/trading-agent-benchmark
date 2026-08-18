import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2033-04-14')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); P[s]=d.set_index('date').close
P=pd.DataFrame(P).sort_index().loc[:cutoff]; R=P.pct_change()
csdisp=R.rolling(20,min_periods=10).std().mean(axis=1,skipna=True)
gate=(csdisp < csdisp.rolling(120,min_periods=60).median()).astype(float)
vol=R.rolling(20,min_periods=10).std()*np.sqrt(20)
f=R.rolling(5,min_periods=3).sum().div(vol).mul(-1).mul(gate.shift(1),axis=0)
os.makedirs('scripts/artifacts',exist_ok=True); f.to_csv('scripts/artifacts/miner_3_20330414_dispersion_gated_short_reversal_signal.csv',index_label='date')
for h in [5,10,20,30]:
 ic=[]; ns=[]; dates=[]; fr=P.shift(-h)/P-1
 for dt in P.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 x=pd.Series(ic,index=pd.to_datetime(dates)).dropna()
 print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'recent260',round(x.tail(260).mean(),6),round(x.tail(260).mean()/x.tail(260).std(ddof=1),6))
 if h==30: x.to_csv('scripts/artifacts/miner_3_20330414_dispersion_gated_short_reversal_ic.csv',header=['ic'],index_label='date')
print('coverage',round(f.notna().sum().sum()/(len(P)*len(U)),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'range',P.index.min(),P.index.max())
