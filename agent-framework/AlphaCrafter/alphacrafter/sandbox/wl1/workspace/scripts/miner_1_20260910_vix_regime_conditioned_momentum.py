import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close']
 px[a]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# one interpretable idea: medium momentum conditioned on volatility regime;
# trend-following in calm regime and reversal in stressed regime
mom=p/p.shift(20)-1
vz=v/v.rolling(60,min_periods=40).median()-1
f=mom.where(vz<=0, -mom)
# rank-center to remove common component
f=f.sub(f.mean(axis=1),axis=0)
for h in [5,10,20,30]:
 fr=p.shift(-h)/p-1
 ics=[]; ns=[]; turns=[]
 prev=None
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]
  ok=x.notna()&y.notna()
  if ok.sum()>=8:
   ics.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
   rr=x[ok].rank(pct=True)
   if prev is not None: turns.append(np.abs(rr-prev).mean())
   prev=rr
 ic=np.nanmean(ics); sd=np.nanstd(ics,ddof=1); ir=ic/sd*np.sqrt(252/h) if sd else np.nan
 print(h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f turnover %.6f'%(ic,ir,np.mean(np.array(ics)>0),len(ics),np.mean(ns),np.mean(turns)))
# annual h20
fr=p.shift(-20)/p-1
for yr in range(2020,2027):
 vals=[]
 for dt in f.index[f.index.year==yr]:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt][ok],fr.loc[dt][ok]).statistic)
 print('year',yr,'IC',round(float(np.nanmean(vals)),6),'n',len(vals))
print('cutoff',p.index.max(),'dates',len(p),'assets',p.shape[1])
