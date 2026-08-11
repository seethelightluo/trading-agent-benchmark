import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close']; return d.loc[:cut]
# Keep each instrument's native calendar before cross-sectional alignment.
R=pd.concat({a:load(a).pct_change(fill_method=None) for a in U},axis=1).sort_index()
vol=R.rolling(20,min_periods=15).std()*np.sqrt(252)
F=(-R.rolling(5,min_periods=5).sum()/vol).shift(1); Y={h:R.shift(-h,fill_value=np.nan) for h in [1,5,10]}
for h in [1,5,10]:
 vals=[]; ds=[]; ns=[]
 for dt in F.index.intersection(Y[h].index):
  z=pd.concat([F.loc[dt],Y[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 a=np.asarray(vals); print('horizon',h,'dates',len(a),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 if h==1:
  print('coverage %.4f turnover %.4f'%(F.stack().notna().mean(),F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
  for y in range(2020,2027):
   v=a[[d.year==y for d in ds]]; print('regime',y,len(v),'IC %.6f ICIR %.6f'%(v.mean(),v.mean()/v.std(ddof=1)) if len(v)>1 else 'NA')
  for n in [252,504,756]:
   v=a[-n:]; print('recent',n,'IC %.6f ICIR %.6f'%(v.mean(),v.mean()/v.std(ddof=1)))
