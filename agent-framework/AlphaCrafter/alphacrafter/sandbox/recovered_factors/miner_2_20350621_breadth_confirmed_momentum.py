import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in A}
C=pd.DataFrame({a:d.close.astype(float) for a,d in D.items()})
R=C.pct_change(); V=R.rolling(20,min_periods=12).std()
# Macro-breadth-confirmed momentum: relative 20-session risk-adjusted trend,
# amplified only when the cross-asset 5-session breadth confirms its direction.
breadth=np.sign(R.rolling(5,min_periods=4).sum()).mean(axis=1)
confirm=(1+0.75*breadth.abs())
F=(R.rolling(20,min_periods=15).sum()/V.replace(0,np.nan))*confirm.values[:,None]
# Remove common cross-asset component, then lag to avoid look-ahead.
F=F.sub(F.median(axis=1),axis=0).shift(1)
F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0)
print('idea=5d-breadth-confirmed 20d risk-adjusted relative momentum; rows=%d assets=%d cells=%d coverage=%.4f turnover=%.4f'%(len(C),len(A),F.notna().sum().sum(),F.notna().mean().mean(),F.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean()))
def calc(h):
 fr=R.rolling(h,min_periods=h).sum().shift(-h); z=[]; n=[]; ds=[]
 for dt in F.index:
  x=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(x)>=8:
   z.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman')); n.append(len(x)); ds.append(dt)
 return np.array(z),n,ds
for h in [1,5,10,20]:
 z,n,ds=calc(h); print('H%d IC %.6f ICIR %.6f dates %d meanN %.2f hit %.4f'%(h,np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1),len(z),np.mean(n),np.mean(z>0)))
 for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-06-06')]:
  q=(pd.Index(ds)>=lo)&(pd.Index(ds)<=hi); zz=z[q]
  print(' ',lo,len(zz),('%.6f %.6f'%(np.nanmean(zz),np.nanmean(zz)/np.nanstd(zz,ddof=1)) if len(zz)>1 else 'NA'))
print('decay complete; exact common-date library correlation audit not completed, so admission would fail absent that evidence')
