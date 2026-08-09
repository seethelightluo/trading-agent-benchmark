import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E='2035-04-25'
def px(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.close,errors='coerce').loc[:E]
P=pd.concat([px(a).rename(a) for a in A],axis=1); R=P.pct_change(); eq=R.mean(axis=1)
# Dispersion-conditioned cross-asset reversal: recent 5d move, inverse vol, amplified when cross-sectional dispersion is high.
disp=R.rolling(5).std(axis=1).shift(1)
rv=R.rolling(20).std().shift(1)
F=(-R.rolling(5).sum().shift(1)).div(rv).mul((disp/disp.rolling(60).median()).clip(.5,2),axis=0)
F=F.replace([np.inf,-np.inf],np.nan)
for h in [1,5,10,20]:
 fr=P.pct_change(h).shift(-h); vals=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(vals); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0),len(x),np.mean(ns)))
for lo,hi in [('2020','2025'),('2025','2030'),('2030','2033'),('2033','2035')]:
 x=[];fr=P.pct_change(10).shift(-10)
 for d in F.loc[lo:hi].index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('REG',lo,hi,'n',len(x),'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1) if len(x)>1 else np.nan)
print('cells',F.count().sum(),'coverage',F.count().sum()/(len(F)*15),'rows',len(P),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
print('cutoff',E)
