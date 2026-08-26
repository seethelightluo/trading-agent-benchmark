import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:end] for s in U}
R=pd.concat({s:D[s].close.pct_change() for s in U},axis=1).sort_index()
# Relative acceleration: recent 5d return minus prior 5d return, cross-sectionally demeaned.
F=R.rolling(5,min_periods=5).sum()-R.shift(5).rolling(5,min_periods=5).sum()
F=F.sub(F.mean(axis=1),axis=0)
for h in [1,5,10]:
  ic=[]; ns=[]; regimes={}
  for dt in F.index:
    xs=[]; ys=[]
    for s in U:
      if pd.isna(F.loc[dt,s]) or dt not in D[s].index: continue
      i=D[s].index.get_loc(dt)
      if i+h<len(D[s]):
        y=D[s].close.iloc[i+h]/D[s].close.iloc[i]-1
        if pd.notna(y): xs.append(F.loc[dt,s]);ys.append(y)
    if len(xs)>=8 and len(set(xs))>1:
      q=spearmanr(xs,ys).statistic
      if pd.notna(q): ic.append(q); ns.append(len(xs)); regimes.setdefault(dt.year,[]).append(q)
  a=np.asarray(ic); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4),'years',{k:round(np.mean(v),5) for k,v in regimes.items()})
print('coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
