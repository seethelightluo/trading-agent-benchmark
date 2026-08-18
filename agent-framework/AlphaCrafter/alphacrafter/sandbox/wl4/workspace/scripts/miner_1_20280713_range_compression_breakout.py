import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
cut='2028-07-13'; ds={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in ['DXY','USDCNY','USDJPY','EURUSD','VIX']: continue
 d=pd.read_csv(f);d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date'); r=d.close.pct_change()
 # breakout signal: medium momentum, strengthened by compressed realized range
 vol20=r.rolling(20,min_periods=15).std(); vol60=r.rolling(60,min_periods=40).std()
 d['f']=d.close.pct_change(20)*(vol60/vol20).clip(0.5,2.0)
 ds[s]=d.set_index('date')[['close','f']]
dates=sorted(set.intersection(*[set(x.index) for x in ds.values()]))
for h in [1,5,10,20]:
 out=[]
 for dt in dates:
  a=[];b=[]
  for s,d in ds.items():
   i=d.index.get_loc(dt)
   if i+h>=len(d):continue
   if pd.notna(d.iloc[i].f):a.append(d.iloc[i].f);b.append(d.iloc[i+h].close/d.iloc[i].close-1)
  if len(a)>=8 and np.std(a)>0 and np.std(b)>0:out.append(spearmanr(a,b).statistic)
 z=pd.Series(out).dropna();print(h,len(z),len(ds),z.mean(),z.mean()/z.std(),(z>0).mean(), 'recent',z.tail(250).mean(),z.tail(250).mean()/z.tail(250).std())
