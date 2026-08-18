import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}
for s in S:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150:d=get_index_daily_data(s,4000)
 if d is not None:F[s]=d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close
p=pd.DataFrame(F).sort_index(); r=p.pct_change(); x=pd.read_csv('../persistent/index_data/VIX.csv');x.date=pd.to_datetime(x.date);v=x.set_index('date').close.reindex(p.index).ffill(); shock=(v.pct_change(5)).rolling(60,min_periods=30).rank(pct=True).clip(.1,.9)
# Contrarian response to recent asset move, magnified after unusually sharp VIX changes.
f=(-p.pct_change(5).mul(0.5+shock,axis=0)).clip(-.5,.5)
for h in [1,5,10,20]:
 z=[]; ns=[]; cov=[]
 for dt in f.index:
  a=f.shift(1).loc[dt];b=p.pct_change(h).shift(-h).loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(a[ok].corr(b[ok],method='spearman'));ns.append(ok.sum());cov.append(ok.mean())
 z=pd.Series(z).dropna();print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.5f} ICIR={z.mean()/z.std(ddof=1):.5f} hit={(z>0).mean():.3f} cov={np.mean(cov):.4f}')
 for n in [250,500]:
  q=z.tail(n);print(f'recent{n} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f}')
print('instruments',len(F),'range',p.index.min().date(),p.index.max().date())
