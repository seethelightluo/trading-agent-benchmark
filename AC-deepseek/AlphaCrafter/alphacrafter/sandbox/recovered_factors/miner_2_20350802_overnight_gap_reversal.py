import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-08-01')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:E] for a in A}
O=pd.concat({a:D[a]['open'] for a in A},axis=1); C=pd.concat({a:D[a]['close'] for a in A},axis=1)
# Lagged overnight-gap pressure reversal, robust to asset scale via open/previous-close return.
gap=O/C.shift(1)-1
sig=gap.rolling(10,min_periods=7).mean().shift(1)
S=-sig.sub(sig.median(axis=1),axis=0); S=S.clip(S.quantile(.1,axis=1),S.quantile(.9,axis=1),axis=0)
for h in [1,5,10,20]:
 f=C.shift(-h)/C-1; vals=[]; dates=[]; ns=[]
 for dt in S.index:
  x=S.loc[dt]; y=f.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); ns.append(ok.sum())
 v=np.array(vals); print('H%d IC %.6f ICIR %.6f dates %d hit %.4f meanN %.2f'%(h,np.nanmean(v),np.nanmean(v)/np.nanstd(v,ddof=1),len(v),np.mean(v>0),np.mean(ns)))
for lo,hi in [('2025','2029'),('2030','2032'),('2033','2035')]:
 x=[]; y=[]
 for dt in S.loc[lo:hi].index:
  ok=S.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:x.append(spearmanr(S.loc[dt][ok],f.loc[dt][ok]).statistic)
 print('REGIME',lo,hi,len(x),np.mean(x) if x else np.nan)
print('rows',len(C),'assets',len(A),'cells',int(S.notna().sum().sum()),'coverage',S.notna().sum().sum()/(len(C)*len(A)))
print('turnover',np.nanmean([np.mean(abs(S.iloc[i].rank(pct=True)-S.iloc[i-1].rank(pct=True))) for i in range(1,len(S))]))
