import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT='../persistent/stock_data'; MAC='../persistent/index_data'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(path):
 d=pd.read_csv(path); d.date=pd.to_datetime(d.date); return d.set_index('date').close.astype(float)
px=pd.DataFrame({a:load(f'{ROOT}/{a}.csv') for a in assets}).sort_index()
vix=load(f'{MAC}/VIX.csv').reindex(px.index).ffill()
# Fade lagged 10d asset moves only after an unusually large VIX increase.
r10=px.pct_change(10); shock=vix.pct_change(3)
threshold=shock.abs().rolling(252,min_periods=126).quantile(.70)
factor=(-r10).where(shock > threshold, 0.0)
fwd=px.shift(-10)/px-1
rows=[]
for dt in px.index:
 x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  rows.append((dt,spearmanr(x[ok],y[ok]).statistic,int(ok.sum())))
a=np.array([r[1] for r in rows]); n=np.array([r[2] for r in rows])
print('dates',len(a),'avg_n',n.mean(),'coverage',np.mean(n/15))
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252),'hit',np.mean(a>0))
for yr in range(2026,2032):
 z=np.array([r[1] for r in rows if r[0].year==yr]); print('year',yr,'dates',len(z),'IC',z.mean() if len(z) else np.nan)
q=factor.replace(0,np.nan).rank(axis=1,pct=True)
print('turnover',q.diff().abs().mean().mean(),'active_fraction',np.mean((factor!=0).sum(axis=1)/15))
for h in [5,10,20]:
 fwdh=px.shift(-h)/px-1; vals=[]
 for dt in px.index:
  x=factor.loc[dt]; y=fwdh.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1: vals.append(spearmanr(x[ok],y[ok]).statistic)
 z=np.array(vals); print('horizon',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(252))
