import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+a+'.csv'
 x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); x=x.set_index('date').sort_index()
 D[a]=x.close.astype(float)
px=pd.DataFrame(D).sort_index(); ret=px.pct_change()
# volatility shock factor: low recent volatility relative to medium-term volatility
v5=ret.rolling(5).std(); v60=ret.rolling(60).std()
f=-(np.log((v5+1e-8)/(v60+1e-8))) # positive when compressed
# lag signal one day, evaluate forward h-day compounded returns
f=f.shift(1)
print('range',px.index.min(),px.index.max(),'cells',f.notna().sum().sum())
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1
 vals=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=dates); print(h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(ns)))
 for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
  q=s[(s.index>=lo)&(s.index<=hi+'-12-31')]; print(' ',lo,hi, '%.6f %.6f %d'%(q.mean(),q.mean()/q.std(),len(q)))
# 10d turnover rank changes
r=f.rank(axis=1,pct=True); print('turn10', (r.diff(10).abs().mean(axis=1).mean()))
print('coverage',f.notna().sum().sum()/f.size)
