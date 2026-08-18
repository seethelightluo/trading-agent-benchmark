import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-04-10'); base=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:end].ffill(); R=P.pct_change(); v=R.rolling(20).std()
r3=P.pct_change(3); r5=P.pct_change(5)
# Smoothed inverse-vol relative reversal: average volatility-normalized 3d and 5d relative returns.
f3=-(r3-r3.median(axis=1).values[:,None])/(v+1e-8); f5=-(r5-r5.median(axis=1).values[:,None])/(v+1e-8); f=(f3+f5)/2
y=P.shift(-10)/P-1; aa=[]; nn=[]; dd=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);nn.append(len(z));dd.append(d)
a=np.array(aa);print('dates',len(a),'N',np.mean(nn),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-04-10')]:
 q=(np.array(dd)>=pd.Timestamp(lo))&(np.array(dd)<=pd.Timestamp(hi));b=a[q];print('REG',lo,len(b),b.mean(),b.mean()/b.std(ddof=1))
rk=f.rank(axis=1,pct=True);print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',(rk-rk.shift()).abs().mean(axis=1).dropna().mean())
for h in [1,3,5,10,20]:
 yy=P.shift(-h)/P-1; zics=[]
 for d in f.index:
  z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8:zics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('DECAY',h,np.mean(zics),np.mean(zics)/np.std(zics,ddof=1))
f.to_csv('scripts/miner_3_20280411_smoothed_volnorm_reversal_signal.csv')
