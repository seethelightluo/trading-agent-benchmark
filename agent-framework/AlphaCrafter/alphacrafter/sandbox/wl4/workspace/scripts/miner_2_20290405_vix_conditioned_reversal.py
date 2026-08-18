import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in syms:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date)
  frames[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(frames).sort_index(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# High-volatility macro regimes favor faster cross-sectional snapback; VIX is observation-only.
vix=pd.read_csv('../persistent/index_data/VIX.csv')
vix['date']=pd.to_datetime(vix['date']); vc='close'
v=vix.set_index('date')[vc].reindex(p.index).ffill(); vp=v.rolling(120,min_periods=60).rank(pct=True)
gate=(0.5+vp).clip(.5,1.5)
f=(-p.pct_change(5)/(vol*np.sqrt(5)).mul(gate, axis=0)).replace([np.inf,-np.inf],np.nan).clip(-10,10)
def calc(h):
 fr=f.shift(1); fw=p.pct_change(h).shift(-h); z=[]; cov=[]; ns=[]; tr=[]
 for i,dt in enumerate(fr.index):
  a,b=fr.loc[dt],fw.loc[dt]; ok=a.notna()&b.notna(); n=int(ok.sum())
  if n>=8: z.append(a[ok].corr(b[ok],method='spearman')); cov.append(ok.mean()); ns.append(n)
  if i:
   a0=fr.iloc[i-1]; oo=a.dropna().index.intersection(a0.dropna().index)
   if len(oo)>=8: tr.append(1-a[oo].rank().corr(a0[oo].rank(),method='spearman'))
 z=pd.Series(z).dropna(); print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.5f} ICIR={z.mean()/z.std(ddof=1):.5f} hit={(z>0).mean():.3f} coverage={np.mean(cov):.4f} turnover={np.mean(tr):.5f}')
 for n in (250,500):
  q=z.tail(n); print(f'recent{n} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} dates={len(q)}')
print('instruments',len(frames),'range',p.index.min().date(),p.index.max().date())
for h in [1,5,10,20]: calc(h)
