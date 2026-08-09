import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-08-01')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:E] for a in A}
O=pd.concat({a:D[a]['open'] for a in A},axis=1); C=pd.concat({a:D[a]['close'] for a in A},axis=1); H=pd.concat({a:D[a]['high'] for a in A},axis=1); L=pd.concat({a:D[a]['low'] for a in A},axis=1)
clv=((C-O)/(H-L).replace(0,np.nan)).clip(-1,1)
S=-clv.rolling(10,min_periods=7).mean().shift(1); S=S.sub(S.median(axis=1),axis=0); S=S.clip(S.quantile(.1,axis=1),S.quantile(.9,axis=1),axis=0)
R=C.pct_change()
for h in [1,5,10,20]:
 f=R.rolling(h).sum().shift(-h); z=[]; ns=[]; ds=[]
 for dt in S.index:
  ok=S.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=spearmanr(S.loc[dt][ok],f.loc[dt][ok]).statistic
   if np.isfinite(q): z.append(q);ns.append(ok.sum());ds.append(dt)
 z=np.array(z); print('H%d IC %.6f ICIR %.6f dates %d hit %.4f meanN %.2f'%(h,z.mean(),z.mean()/z.std(ddof=1),len(z),np.mean(z>0),np.mean(ns)))
 for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-08-01')]:
  q=(pd.Index(ds)>=lo)&(pd.Index(ds)<=hi); zz=z[q]; print(' ',lo,len(zz),round(zz.mean(),6) if len(zz) else None)
print('rows',len(C),'valid_cells',int(S.notna().sum().sum()),'coverage',S.notna().mean().mean(),'turnover',S.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean())
print('library correlation audit: NOT COMPLETED (required evidence absent)')
