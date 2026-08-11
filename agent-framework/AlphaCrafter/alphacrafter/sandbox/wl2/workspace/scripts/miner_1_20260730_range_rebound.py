import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close']; return d.loc[:cut]
P=pd.concat({a:load(a) for a in U},axis=1).sort_index(); R=P.pct_change(fill_method=None)
# Distance from trailing 60-day high/low, with low values favored as rebound candidates.
hi=P.rolling(60,min_periods=40).max(); lo=P.rolling(60,min_periods=40).min(); F=((P-lo)/(hi-lo)).clip(0,1).shift(1); Y=R.shift(-1)
for h in [1,5,10]:
 Yh=P.pct_change(h).shift(-h)
 vals=[];ds=[];ns=[]
 for dt in F.index.intersection(Yh.index):
  z=pd.concat([F.loc[dt],-Yh.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'N',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 if h==1:
  print('coverage',F.stack().notna().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
  for y in range(2020,2027):
   v=a[[d.year==y for d in ds]]; print(y,len(v),v.mean() if len(v) else np.nan,v.mean()/v.std(ddof=1) if len(v)>1 else np.nan)
