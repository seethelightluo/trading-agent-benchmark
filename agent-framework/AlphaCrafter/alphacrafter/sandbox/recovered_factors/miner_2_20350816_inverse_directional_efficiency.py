import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-08-15')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:E] for a in A}
C=pd.DataFrame({a:pd.to_numeric(D[a]['close'],errors='coerce') for a in A}); R=C.pct_change(fill_method=None)
# One interpretable idea: inverse directional efficiency. Persistent movement with a large net move
# is treated as overextended; choppy paths receive less reversal signal. Lag one completed day.
abs_sum=R.abs().rolling(10,min_periods=7).sum(); net=R.rolling(10,min_periods=7).sum()
F=(-net/(abs_sum+1e-12)).shift(1)
F=F.sub(F.median(axis=1),axis=0); F=F.clip(F.quantile(.1,axis=1),F.quantile(.9,axis=1),axis=0)
print('candidate inverse directional efficiency reversal: rows=%d assets=%d valid_cells=%d coverage=%.4f meanN=%.2f turnover=%.6f'%(len(C),len(A),int(F.notna().sum().sum()),F.notna().mean().mean(),F.notna().sum(axis=1).mean(),F.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean()))
for h in [1,5,10,20]:
 fw=C.shift(-h)/C-1; z=[]; ns=[]; ds=[]
 for dt in F.index:
  q=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.r.nunique()>1:
   z.append(spearmanr(q.f,q.r).statistic); ns.append(len(q)); ds.append(dt)
 z=np.asarray(z); ds=pd.DatetimeIndex(ds); ir=z.mean()/z.std(ddof=1)
 print('H%d IC %.6f ICIR %.6f dates %d hit %.4f meanN %.2f se %.6f'%(h,z.mean(),ir,len(z),np.mean(z>0),np.mean(ns),z.std(ddof=1)/np.sqrt(len(z))))
 for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-08-15')]:
  q=(ds>=lo)&(ds<=hi); zz=z[q]; print(' REG',lo,'n',len(zz),'IC',f'{zz.mean():.6f}' if len(zz) else 'nan','hit',f'{np.mean(zz>0):.4f}' if len(zz) else 'nan')
print('decay above; all dates point-in-time through',E.date())
# A conservative machine-checkable partial audit against a few known proxy families; not admission evidence.
proxy={'momentum_20':R.rolling(20,min_periods=15).sum(),'reversal_5':-R.rolling(5,min_periods=4).sum(),'volatility_20':R.rolling(20,min_periods=15).std()}
for n,x in proxy.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); print('proxy_corr',n,f'{q.f.corr(q.x,method="spearman"):.6f}',len(q))
print('REQUIRED full admitted-library correlation audit: NOT COMPLETED; no persistence permitted without exact max_abs_library_correlation')
print('history_start',C.index.min().date(),'history_end',C.index.max().date(),'usable_assets',len(A))
