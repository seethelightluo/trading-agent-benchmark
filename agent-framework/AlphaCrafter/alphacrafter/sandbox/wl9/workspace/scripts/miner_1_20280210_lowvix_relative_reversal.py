import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-09'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(5)
vix=pd.read_csv(Path('../persistent/index_data/VIX.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end]
vix=vix.reindex(px.index).ffill(); med=vix.rolling(60,min_periods=30).median()
# cross-sectional relative 5d reversal, active in subdued-volatility regimes
fac=-(r.sub(r.mean(axis=1),axis=0)).where(vix.lt(med),np.nan)
print('candidate low-VIX relative 5d reversal')
for h in [1,3,5,10]:
 fwd=px.shift(-h)/px-1; out=[]; dates=[]; counts=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); counts.append(len(z))
 a=np.asarray(out); print('horizon',h,'dates',len(a),'avg_n',np.mean(counts),'mean_ic',a.mean(),'icir',a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),'hit',(a>0).mean())
# regime slices for admission horizon 5
fwd=px.shift(-5)/px-1; out=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.Series(dict(out));
for label,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026+',a.index>='2026-01-01'),('2027+',a.index>='2027-01-01')]:
 q=a[mask]; print(label,'dates',len(q),'ic',q.mean(),'icir',q.mean()/(q.std(ddof=1)/np.sqrt(len(q))) if len(q)>1 else np.nan)
print('coverage',fac.notna().mean().mean(),'active_fraction',fac.notna().any(axis=1).mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',px.index.min().date(),px.index.max().date())
