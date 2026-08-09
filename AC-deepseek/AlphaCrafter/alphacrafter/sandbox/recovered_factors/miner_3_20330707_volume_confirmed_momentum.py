import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 d=pd.read_csv(f); d.date=pd.to_datetime(d.date); P[a]=d.set_index('date')
C=pd.DataFrame({a:P[a].close for a in A}).sort_index(); V=pd.DataFrame({a:P[a].volume.replace(0,np.nan) for a in A}).sort_index(); R=C.pct_change()
# volume-confirmed momentum: risk-adjusted 20d return, continuously gated by signed volume participation
mom=C.pct_change(20)/(R.rolling(20).std()+1e-12)
vol=np.log(V/(V.rolling(20).median()+1e-12)).clip(-3,3)
sig=(mom*np.tanh(vol/1.5)).shift(1)
print('dates',C.index.min(),C.index.max(),'assets',len(A),'coverage',sig.notna().sum().sum()/(sig.size))
allz={}
for h in [1,5,10,20]:
 f=C.shift(-h)/C-1; z=[]; ds=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   z.append(spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic);ds.append(dt);ns.append(ok.sum())
 z=np.asarray(z); allz[h]=(z,ds)
 print('H%d dates %d meanN %.2f IC %.6f ICIR %.6f hit %.4f'%(h,len(z),np.mean(ns),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),np.mean(z>0)))
 for lo,hi in [('2024','2027'),('2028','2030'),('2031','2033')]:
  q=z[(np.array(ds)>=pd.Timestamp(lo+'-01-01'))&(np.array(ds)<=pd.Timestamp(hi+'-12-31'))];print(' ',lo,len(q),'IC %.6f ICIR %.6f'%(np.mean(q),np.mean(q)/np.std(q,ddof=1)))
print('turn10',sig.rank(pct=True).diff(10).abs().mean(axis=1).dropna().mean())
# correlation with plain risk adjusted momentum signal across all valid cells
base=mom.shift(1); pairs=[]
for dt in sig.index:
 ok=sig.loc[dt].notna()&base.loc[dt].notna()
 if ok.sum()>=8:pairs.append(spearmanr(sig.loc[dt,ok],base.loc[dt,ok]).statistic)
print('daily rank corr vs mom: mean',np.mean(pairs),'maxabs',np.max(np.abs(pairs)))
