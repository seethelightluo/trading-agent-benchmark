import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d=d[pd.to_datetime(d.date)<='2029-04-04']; d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); fs[a]=d
for look in [10,20,40,60]:
 out=[]
 for a,d in fs.items():
  r=d.close.pct_change(); f=r.rolling(look,min_periods=look).sum().shift(1)
  for h in [1,3,5,10]:
   fr=d.close.pct_change(h).shift(-h)
   out += [(dt,a,f.loc[dt],fr.loc[dt],h) for dt in d.index]
 x=pd.DataFrame(out,columns=['date','a','f','r','h']).dropna()
 print('LOOK',look)
 for h,g0 in x.groupby('h'):
  z=[]
  for dt,g in g0.groupby('date'):
   if len(g)>=8:z.append(spearmanr(g.f,g.r).statistic)
  z=np.array(z);print(h,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1))
