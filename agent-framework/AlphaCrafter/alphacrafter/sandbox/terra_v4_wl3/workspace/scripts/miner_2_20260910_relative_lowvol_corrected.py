import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base=Path('../persistent/stock_data'); end='2026-09-09'
def load(s):
 d=pd.read_csv(base/(s+'.csv')); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').close.astype(float).sort_index()
px=pd.DataFrame({s:load(s) for s in U}).sort_index().loc[:end]
r=px.pct_change(fill_method=None)
# Explicit per-asset session rolling, then reindex: avoids union-calendar NaN contamination.
vol=pd.DataFrame({s:r[s].dropna().rolling(20,min_periods=20).std() for s in U})
cs_med=vol.median(axis=1); f=-vol.div(cs_med,axis=0)
def ev(h):
 y=pd.DataFrame({s:px[s].dropna().pct_change(h).shift(-h) for s in U})
 aa=[]; dd=[]; nn=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):aa.append(q);dd.append(d);nn.append(len(z))
 return pd.Series(aa,index=dd),nn
print('candidate=relative_low_vol_20d_exact; assets',len(U),'rows',len(px))
for h in [1,5,10]:
 a,n=ev(h); print('h',h,'dates',len(a),'avg_names',round(np.mean(n),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
a,n=ev(1); rank=f.rank(axis=1,pct=True)
print('coverage',round(f.notna().sum().sum()/f.size,4),'factor_dates',f.notna().any(axis=1).sum(),'turnover',round(rank.diff().abs().mean().mean(),4))
print('year',a.groupby(a.index.year).agg(['mean','count']).round(5).to_dict())
f.loc[a.index].to_csv('scripts/miner_2_20260910_relative_lowvol_signal.csv')
