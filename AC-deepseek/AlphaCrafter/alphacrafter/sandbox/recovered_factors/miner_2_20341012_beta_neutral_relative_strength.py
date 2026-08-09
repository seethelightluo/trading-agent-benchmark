import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-10-11')
px={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); px[a]=d.loc[d.index<=E,'close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(fill_method=None)
# Candidate: beta-neutral medium-horizon relative strength. Remove each asset's
# rolling 60d beta to the equal-weight cross-asset return, then rank residual 20d return.
mkt=R.mean(axis=1); mw=mkt.rolling(60,min_periods=40).var()
beta=R.rolling(60,min_periods=40).cov(mkt).div(mw,axis=0)
res=R.sub(beta.mul(mkt,axis=0))
F=res.rolling(20,min_periods=15).sum()
print('candidate beta-neutral relative strength; cutoff',E.date(),'rows',len(P),'assets',len(A))
all_ic={}
for h in [1,5,10,20]:
 vals=[];ns=[];dates=[]; fr=P.shift(-h)/P-1
 for t in F.index:
  z=pd.concat([F.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(t)
 q=np.asarray(vals);ds=pd.Series(dates);all_ic[h]=(q,ds)
 print('H',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1),np.mean(q>0)))
print('coverage %.6f turnover %.6f'%(F.notna().mean().mean(),F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for lo,hi in [('2020-01-01','2024-12-31'),('2025-01-01','2029-12-31'),('2030-01-01','2034-10-11'),('2033-01-01','2034-10-11')]:
 q,ds=all_ic[10];m=(ds>=lo)&(ds<=hi);y=q[m]
 print('regime',lo,hi,'dates',len(y),'IC %.6f ICIR %.6f'%(np.nanmean(y),np.nanmean(y)/np.nanstd(y,ddof=1)) if len(y)>1 else 'nan')
print('library_audit FAILED: exact common-cell values for all admitted factor expressions unavailable; max_abs_library_correlation missing')
