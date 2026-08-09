import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]
 if s in keep:
  x=pd.read_csv(f);x.date=pd.to_datetime(x.date); y=x.set_index('date'); d[s]=y[['close','volume']]
cl=pd.DataFrame({s:x.close for s,x in d.items()}).sort_index().loc[:'2033-03-30']
vo=pd.DataFrame({s:x.volume for s,x in d.items()}).reindex(cl.index)
r=cl.pct_change(); vr=vo/vo.rolling(20,min_periods=10).median()
# Volume-confirmed medium momentum: 10d return weighted by persistent volume surprise, lagged one day.
f=(r.rolling(10,min_periods=8).sum()*vr.rolling(5,min_periods=3).mean().clip(0.5,3)).shift(1)
print('candidate volume_confirmed_momentum_10d; dates',len(cl),'assets',len(cl.columns))
for h in [1,5,10,20]:
 fr=cl.shift(-h)/cl-1;a=[];ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.array(a);print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for a,b in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 z=[]
 for dt in f.loc[a:b].index:
  q=pd.concat([f.loc[dt],(cl.shift(-10)/cl-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print(a,b,'regime10',len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
