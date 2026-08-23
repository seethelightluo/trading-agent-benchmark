import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT='../persistent/stock_data'; MAC='../persistent/index_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(path):
 d=pd.read_csv(path); d.date=pd.to_datetime(d.date); return d.set_index('date').close.astype(float)
p={a:load(f'{ROOT}/{a}.csv') for a in assets}; px=pd.DataFrame(p).sort_index(); dxy=load(f'{MAC}/DXY.csv').reindex(px.index).ffill()
# macro-conditioned cross-sectional reversal: when DXY has a 5d positive shock, fade each asset's lagged 10d return;
# otherwise use zero signal. Signal is ranked cross-sectionally, and only lagged observations are used.
r10=px.pct_change(10); shock=dxy.pct_change(5)
# use DXY shock known at t, factor at t is applied to t+1 return; condition threshold based on rolling historical abs percentile
thr=shock.abs().rolling(252,min_periods=126).quantile(.70)
factor=(-r10).where(shock > thr, 0.0)
fwd=px.shift(-10)/px-1
rows=[]; daily=[]
for dt in px.index:
 x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  ic=spearmanr(x[ok],y[ok]).statistic
  rows.append((dt,ic,int(ok.sum())))
  daily.append(ic)
arr=np.array(daily); print('dates',len(arr),'avg_n',np.mean([r[2] for r in rows]),'coverage',np.mean([r[2]/15 for r in rows]))
print('IC',arr.mean(),'ICIR',arr.mean()/arr.std(ddof=1)*np.sqrt(252),'hit',np.mean(arr>0))
for yr in range(2026,2032):
 z=np.array([r[1] for r in rows if r[0].year==yr]); print(yr,len(z),z.mean() if len(z) else np.nan)
# signal turnover among nonzero condition dates
q=factor.replace(0,np.nan).rank(axis=1,pct=True); print('turnover',q.diff().abs().mean().mean(),'nonzero',np.mean((factor!=0).sum(axis=1)/15))
