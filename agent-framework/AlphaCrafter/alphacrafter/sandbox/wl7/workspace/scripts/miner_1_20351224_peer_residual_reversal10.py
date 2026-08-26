import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
P=pd.concat({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index()
r=P.pct_change(); raw=r.rolling(10).sum(); peer=raw.sub(raw.median(axis=1),axis=0)
# Contrarian cross-sectional residual shock, lagged one day
fac=(-peer).shift(1)
rows_by={}
for h in [1,5,10,20]:
 rows=[]
 for i in range(len(P)-h):
  x=fac.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: rows.append((P.index[i],ok.sum(),x[ok].corr(y[ok],method='spearman')))
 q=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); ir=q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252)
 print(f'H{h} IC {q.ic.mean():.8f} ICIR {ir:.8f} hit {(q.ic>0).mean():.4f} dates {len(q)} avgN {q.n.mean():.2f}')
 for label,lo,hi in [('2020-29',2020,2030),('2030-34',2030,2035),('2035',2035,2036)]:
  z=q[(q.date.dt.year>=lo)&(q.date.dt.year<hi)]
  if len(z): print(' ',label,round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252),4),len(z))
print('assets',P.shape[1],'dates',len(P),'coverage',fac.notna().mean().mean(),'turnover',np.nanmean(np.abs(np.diff(np.nan_to_num(fac.values,nan=0),axis=0))))
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20351224_peer_residual_reversal10_signal.csv',index=False)
print('artifact_rows',len(out))
