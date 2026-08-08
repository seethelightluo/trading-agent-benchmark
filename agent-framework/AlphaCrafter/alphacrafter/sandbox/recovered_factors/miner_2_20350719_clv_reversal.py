import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-07-18')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:E] for a in A}
O=pd.concat({a:D[a]['open'] for a in A},axis=1); C=pd.concat({a:D[a]['close'] for a in A},axis=1); H=pd.concat({a:D[a]['high'] for a in A},axis=1); L=pd.concat({a:D[a]['low'] for a in A},axis=1)
# Persistent close-location pressure: average daily CLV, scaled by its own recent volatility, lagged one day.
clv=((C-O)/(H-L).replace(0,np.nan)).clip(-1,1)
sig=clv.rolling(10,min_periods=7).mean().shift(1)
# reversal of persistent selling pressure, cross-sectional demean and winsorize
S=-sig.sub(sig.median(axis=1),axis=0); S=S.clip(S.quantile(.1,axis=1),S.quantile(.9,axis=1),axis=0)
for h in [1,5,10,20]:
 f=C.shift(-h)/C-1; z=[]; ds=[]; ns=[]
 for dt in S.index:
  ok=S.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8: z.append(spearmanr(S.loc[dt][ok],f.loc[dt][ok]).statistic);ds.append(dt);ns.append(ok.sum())
 z=np.array(z); print('H%d IC %.6f ICIR %.6f dates %d hit %.4f meanN %.2f'%(h,np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1)*np.sqrt(252/h),len(z),np.mean(z>0),np.mean(ns)))
 for lo,hi in [('2025','2029'),('2030','2032'),('2033','2035')]:
  q=np.array([v for v,d in zip(z,ds) if lo<=str(d.year)<=hi]); print(lo+'-'+hi,len(q),f'{np.mean(q) if len(q) else np.nan:.6f}',f'{np.mean(q)/np.std(q,ddof=1)*np.sqrt(252/h) if len(q)>1 else np.nan:.6f}')
print('rows',len(C),'assets',len(A),'cells',S.notna().sum().sum(),'coverage',S.notna().sum().sum()/S.size,'turnover',S.rank(pct=True).diff().abs().mean(axis=1).mean())
