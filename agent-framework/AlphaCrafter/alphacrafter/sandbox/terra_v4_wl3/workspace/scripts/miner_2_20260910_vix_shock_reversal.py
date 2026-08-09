import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); macro=Path('../persistent/index_data')
def load(p):
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); return d.drop_duplicates('date').set_index('date')['close'].astype(float).sort_index()
series={s:load(base/(s+'.csv')) for s in U}; v=load(macro/'VIX.csv')
# exact per-series construction avoids union-calendar rolling artifacts
shock=v.pct_change(fill_method=None).rolling(5,min_periods=5).sum().clip(-1,1)
f={}
for s,p in series.items():
 r=p.pct_change(fill_method=None); f[s]=(-r.rolling(10,min_periods=10).sum()).mul(1+shock.reindex(p.index))
f=pd.DataFrame(f).sort_index().loc[:'2026-09-09']; px=pd.DataFrame(series).sort_index().loc[:'2026-09-09']
def calc(h):
 y=px.pct_change(h,fill_method=None).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); dates.append(dt); ns.append(len(z))
 return pd.Series(vals,index=pd.DatetimeIndex(dates)),ns
print('candidate=vix_shock_conditioned_reversal_10d'); print('range',f.index.min(),f.index.max(),'assets',len(U),'rows',len(f))
for h in [1,5,10]:
 a,n=calc(h); print('h',h,'dates',len(a),'avg_names',round(np.mean(n),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
a,n=calc(1); print('coverage',round(f.notna().sum().sum()/f.size,4),'rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('regime',a.groupby(shock.reindex(a.index).gt(0).map({True:'vix_rising',False:'vix_falling'})).agg(['mean','count']).round(5).to_dict())
print('year',a.groupby(a.index.year).agg(['mean','count']).round(5).to_dict())
