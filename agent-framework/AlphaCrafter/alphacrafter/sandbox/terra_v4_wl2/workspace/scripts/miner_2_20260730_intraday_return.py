import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
# Prior-session intraday return, a distinct microstructure/overnight decomposition signal.
F={}; R={}
for s in U:
 d=D[s]; F[s]=(d.close/d.open-1).replace([np.inf,-np.inf],np.nan); R[s]=d.close.pct_change()
FI=pd.concat(F,axis=1); results={}
for h in [1,5,10]:
  ics=[]; ns=[]; years={}
  for dt in FI.index:
    xs=[]; ys=[]
    for s in U:
      if dt not in D[s].index or pd.isna(F[s].get(dt)): continue
      ix=D[s].index.get_loc(dt); j=ix+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]): xs.append(F[s].loc[dt]); ys.append(D[s].close.iloc[j]/D[s].close.iloc[ix]-1)
    if len(xs)>=8 and len(set(xs))>1:
      z=spearmanr(xs,ys).statistic; ics.append(z); ns.append(len(xs)); years.setdefault(dt.year,[]).append(z)
  a=np.array(ics); results[h]=(len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),{k:np.mean(v) for k,v in years.items()})
for h,x in results.items(): print('h',h,'dates',x[0],'meanN',round(x[1],2),'IC',round(x[2],5),'ICIR',round(x[3],5),'hit',round(x[4],4),'years', {k:round(v,4) for k,v in x[5].items()})
print('coverage',round(FI.notna().sum().sum()/(len(FI)*15),4))
r=FI.rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean(axis=1).mean()*2,4))
print('raw_cross_asset_corr_to_existing_like')
for name,p in [('rev5',pd.concat({s:-R[s].rolling(5).sum() for s in U},axis=1)),('mom20',pd.concat({s:R[s].rolling(20).sum() for s in U},axis=1))]: print(name,round(FI.stack().corr(p.stack()),4))
