import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data,get_account_dict
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in syms:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); fs[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(fs).sort_index(); r=p.pct_change();
# Relative strength: each asset's 30d return minus contemporaneous cross-sectional median.
raw=p.pct_change(30); f=raw.sub(raw.median(axis=1),axis=0).clip(-3,3).shift(1)
def calc(h):
 fw=p.pct_change(h).shift(-h); z=[]; cov=[]; ns=[]; tr=[]
 for i,dt in enumerate(f.index):
  a=f.loc[dt]; b=fw.loc[dt]; ok=a.notna()&b.notna(); n=int(ok.sum())
  if n>=8: z.append(a[ok].corr(b[ok],method='spearman')); cov.append(ok.mean()); ns.append(n)
  if i:
   q=a.dropna().index.intersection(f.iloc[i-1].dropna().index)
   if len(q)>=8: tr.append(1-a[q].rank().corr(f.iloc[i-1][q].rank(),method='spearman'))
 z=pd.Series(z).dropna(); print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.5f} ICIR={z.mean()/z.std(ddof=1):.5f} hit={(z>0).mean():.3f} coverage={np.mean(cov):.4f} turnover={np.mean(tr):.5f}')
 for n in (250,500):
  q=z.tail(n); print(f'recent{n} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} dates={len(q)}')
for h in [1,5,10,20]: calc(h)
print('instruments',len(fs),'dates',p.index.min().date(),p.index.max().date())
