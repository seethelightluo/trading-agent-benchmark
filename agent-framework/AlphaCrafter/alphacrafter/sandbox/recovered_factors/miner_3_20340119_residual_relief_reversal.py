import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(a,ix=False):
 p=('../persistent/index_data/' if ix else '../persistent/stock_data/')+a+'.csv'; d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); return d.close.astype(float)
P=pd.DataFrame({a:ld(a) for a in A}).sort_index(); r=np.log(P).diff(); m=r.mean(axis=1)
# residual 10d return: remove common cross-asset movement; favor recent losers only during improving volatility
res=(r.sub(m,axis=0)).rolling(10,min_periods=8).sum()
v=ld('VIX',True).reindex(P.index).ffill(); relief=-(np.log(v).diff(5).rolling(5,min_periods=3).mean())
# smooth relief gate, positive when VIX is declining; avoid scale instability
sig=(-res).mul((1+np.tanh(relief*4)).clip(0.2,1.8),axis=0).shift(1)
print('END',P.index.max(),'cells',sig.notna().sum().sum())
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; z=[]; ns=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8: z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.array(z); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0),len(z),np.mean(ns)))
# regimes H10
h=10; f=P.shift(-h)/P-1; out=[]
for d in sig.index:
 ok=sig.loc[d].notna()&f.loc[d].notna()
 if ok.sum()>=8: out.append((d,spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic))
for a,b in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2034-01-18')]:
 z=np.array([x for d,x in out if d>=pd.Timestamp(a) and d<=pd.Timestamp(b)]); print('REG',a,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
print('coverage',sig.notna().sum().sum()/(len(sig)*15),'turn10',np.mean([np.nanmean((sig.iloc[i].rank(pct=True)-sig.iloc[i-10].rank(pct=True)).abs()) for i in range(10,len(sig))]))
