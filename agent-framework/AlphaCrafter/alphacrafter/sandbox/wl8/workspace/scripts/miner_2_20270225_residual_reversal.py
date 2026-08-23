import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-02-24')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
R=pd.concat({s:D[s].close.pct_change() for s in U},axis=1)
# residual to contemporaneous cross-asset median, smoothed over 3 sessions; negative residual is reversal
M=R.median(axis=1)
F=-(R.sub(M,axis=0)).rolling(3,min_periods=2).sum()
for h in [1,5,10]:
  ics=[]; ns=[]; yrs={}
  for dt in F.index:
    xs=[];ys=[]
    for s in U:
      if pd.isna(F.loc[dt,s]) or dt not in D[s].index: continue
      ix=D[s].index.get_loc(dt); j=ix+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]): xs.append(F.loc[dt,s]);ys.append(D[s].close.iloc[j]/D[s].close.iloc[ix]-1)
    if len(xs)>=8 and len(set(xs))>1:
      q=spearmanr(xs,ys).statistic; ics.append(q);ns.append(len(xs)); yrs.setdefault(str(dt.year),[]).append(q)
  a=np.array(ics); print('h',h,'dates',len(a),'rows',sum(ns),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'years',{k:round(np.mean(x),4) for k,x in yrs.items()})
print('coverage',round(np.mean(F.notna().sum(axis=1)/15),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
