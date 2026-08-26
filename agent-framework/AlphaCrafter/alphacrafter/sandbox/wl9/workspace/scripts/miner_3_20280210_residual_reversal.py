import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-09'); P={}
for s in S:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:end]
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
# 5d return residualized against rolling 60d equal-weight market return; negative residual is reversal
R5=px.pct_change(5); beta=r.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0)
res=R5-beta.mul(m.rolling(5).sum(),axis=0)
fac=-res
# cap extreme observations cross-sectionally to reduce crypto/outlier dominance
fac=fac.clip(lower=fac.quantile(.05,axis=1),upper=fac.quantile(.95,axis=1),axis=0)
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; out=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt)
 a=np.asarray(out); print('horizon',h,'dates',len(a),'assets_mean',round(np.mean([fac.loc[d].notna().sum() for d in dates]),2),'mean_ic',a.mean(),'icir',a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),'hit',(a>0).mean())
# post-online and regime halves
for label, lo, hi in [('early','2020-01-01','2024-01-01'),('late','2024-01-01','2028-02-09'),('online','2026-07-16','2028-02-09')]:
 out=[]; fwd=px.shift(-1)/px-1
 for dt in fac.loc[lo:hi].index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(out); print(label,'dates',len(a),'ic',a.mean() if len(a) else np.nan,'icir',a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else np.nan,'hit',(a>0).mean() if len(a) else np.nan)
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',px.index.min().date(),px.index.max().date())
