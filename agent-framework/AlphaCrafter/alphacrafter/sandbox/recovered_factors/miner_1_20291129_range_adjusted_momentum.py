import pandas as pd, numpy as np, os, json, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; hi={}; lo={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 px[a]=d.close; hi[a]=d.high; lo[a]=d.low
p=pd.DataFrame(px).sort_index(); h=pd.DataFrame(hi).reindex(p.index); l=pd.DataFrame(lo).reindex(p.index)
r=p.pct_change()
# Directional trend normalized by true intraday range: medium momentum divided by recent range volatility.
range_ret=(h-l).div(p.shift(1)).replace([np.inf,-np.inf],np.nan)
f=(r.rolling(20,min_periods=15).sum()/range_ret.rolling(20,min_periods=15).mean()).shift(1)
fr={k:p.shift(-k)/p-1 for k in [1,5,10,20]}
for k,y in fr.items():
 vals=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic); ns.append(ok.sum())
 s=pd.Series(vals); print('h=%d dates=%d meanN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(k,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('coverage=%.6f mean_valid=%.3f turnover10=%.6f'%(f.notna().sum().sum()/f.size,f.notna().sum(axis=1).mean(),f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean()))
for label,start,end in [('2020-24','2020','2024-12-31'),('2025-27','2025','2027-12-31'),('2028-29','2028','2029-11-28')]:
 y=fr[10]; vals=[]
 for dt in f.loc[start:end].index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic)
 s=pd.Series(vals); print('regime=%s n=%d IC=%.6f ICIR=%.6f'%(label,len(s),s.mean(),s.mean()/s.std(ddof=1)))
# comparator audit against current factor signals reconstructed from JSON expressions is not reliable; print files for manual screen
print('library_files',len([x for x in glob.glob('factors/*.json') if not x.endswith('_deprecated')]))
