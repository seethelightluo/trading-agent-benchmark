import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close']
 px[a]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Breadth-conditioned momentum: medium trend is trusted when the cross-asset
# median trend is positive, and inverted when median trend is negative.
# Cross-sectional centering prevents the common market move entering the rank signal.
mom=p/p.shift(10)-1
breadth=mom.median(axis=1,skipna=True)
f=mom.where(breadth>=0,-mom)
f=f.sub(f.mean(axis=1),axis=0)
for h in [5,10,20,30]:
 fr=p.shift(-h)/p-1; ics=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   ics.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
   rr=x[ok].rank(pct=True)
   if prev is not None: turns.append(np.abs(rr-prev.reindex(rr.index)).mean())
   prev=rr
 ic=np.nanmean(ics); sd=np.nanstd(ics,ddof=1); ir=ic/sd*np.sqrt(252/h) if sd else np.nan
 print(f'h={h} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(np.array(ics)>0):.4f} dates={len(ics)} avgN={np.mean(ns):.2f} turnover={np.mean(turns):.6f}')
fr=p.shift(-10)/p-1
for yr in range(2020,2027):
 vals=[]
 for dt in f.index[f.index.year==yr]:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt][ok],fr.loc[dt][ok]).statistic)
 print('year',yr,'IC',f'{np.nanmean(vals):.6f}','n',len(vals))
print('cutoff',p.index.max(),'dates',len(p),'assets',p.shape[1])
