import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-26'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); ret=px.pct_change(); raw=px.pct_change(3)
disp=raw.std(axis=1)
# relative 3d reversal activated only on above-median trailing 60d dispersion
threshold=disp.rolling(60,min_periods=30).median()
fac=-(raw.sub(raw.mean(axis=1),axis=0)).where(disp.gt(threshold), np.nan)
fw=px.shift(-1)/px-1
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; out=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(out); print('horizon',h,'dates',len(a),'assets',len(S),'mean_ic',a.mean(),'icir',a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),'hit',(a>0).mean())
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('period',px.index.min().date(),px.index.max().date())
