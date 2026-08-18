import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2034-08-03')
px={}
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv'); d['date']=pd.to_datetime(d['date']); d=d[d.date<=cutoff].sort_values('date').set_index('date'); px[s]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); pos=r.clip(lower=0).rolling(30,min_periods=15).mean(); down=(-r.clip(upper=0)).pow(2).rolling(30,min_periods=15).mean().pow(.5); f=(pos/(down+1e-8)).shift(1); fr=P.shift(-10)/P-1
ics=[]; counts=[]; cov=[]; turns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): ics.append((dt,q)); counts.append(len(z)); cov.append(len(z)/15)
  turns.append((f.loc[dt].rank(pct=True)-f.shift(1).loc[dt].rank(pct=True)).abs().mean())
I=pd.Series(dict(ics)).sort_index(); print('dates',len(I),'range',I.index.min(),I.index.max(),'avg_n',np.mean(counts),'coverage',np.mean(cov)); print('IC %.6f ICIR %.6f hit %.4f'%(I.mean(),I.mean()/I.std(ddof=1),(I>0).mean()))
for n in [120,260,520,780]:
 q=I.tail(n); print('recent',n,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('turnover',np.nanmean(turns))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'%.6f'%np.nanmean(a),'n',len(a))
