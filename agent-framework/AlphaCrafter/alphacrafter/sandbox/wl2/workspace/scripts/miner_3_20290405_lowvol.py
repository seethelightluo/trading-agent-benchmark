import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d=d[pd.to_datetime(d.date)<='2029-04-04']; d.date=pd.to_datetime(d.date); fs[a]=d.set_index('date').sort_index()
for w in [5,10,20,40]:
 o=[]
 for a,d in fs.items():
  r=d.close.pct_change(); f=-r.rolling(w,min_periods=w).std().shift(1)
  for h in [1,3,5,10]:
   fr=d.close.pct_change(h).shift(-h);o += [(t,a,f.loc[t],fr.loc[t],h) for t in d.index]
 x=pd.DataFrame(o,columns=['date','a','f','r','h']).dropna();print('W',w)
 for h,g0 in x.groupby('h'):
  z=[]
  for t,g in g0.groupby('date'):
   if len(g)>=8:z.append(spearmanr(g.f,g.r).statistic)
  z=np.array(z);print(h,len(z),'IC %.6f IR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
