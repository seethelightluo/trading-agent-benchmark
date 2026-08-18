import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2028-02-10'); base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[s]=d.loc[d.index<=cutoff,'close'].replace(0,np.nan)
P=pd.DataFrame(px); R=P.pct_change(); mom=P/P.shift(10)-1; csmed=mom.median(axis=1); vol=R.rolling(20,min_periods=15).std(); f=mom.sub(csmed,axis=0).div(vol.replace(0,np.nan)); fr=P.shift(-5)/P-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
A=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ics=A.ic.dropna(); mean=ics.mean(); std=ics.std(ddof=1)
print('factor=relative_momentum_10d dates=%d instruments=%d avg_n=%.2f coverage=%.4f'%(len(A),len(U),A.n.mean(),len(A)/len(f)))
print('IC=%.6f ICIR=%.6f hit=%.4f std=%.6f'%(mean,mean/std,(ics>0).mean(),std))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2028-02-10')]:
 q=ics.loc[lo:hi]; print('regime',lo,hi,'n=%d IC=%.6f ICIR=%.6f'%(len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan))
for h in [1,5,10,20]:
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay_%dd=%.6f n=%d'%(h,np.nanmean(vals),len(vals)))
