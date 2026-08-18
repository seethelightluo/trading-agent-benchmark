import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
cut='2028-07-13';ds={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:continue
 d=pd.read_csv(f);d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date');r=d.close.pct_change()
 d['f']=d.close.pct_change(20)/(r.rolling(20,min_periods=15).std()*np.sqrt(20))
 ds[s]=d.set_index('date')[['close','f']]
dates=sorted(set.intersection(*[set(x.index) for x in ds.values()])); out=[]
for dt in dates:
 a=[];b=[]
 for s,d in ds.items():
  i=d.index.get_loc(dt)
  if i+1>=len(d):continue
  v=d.iloc[i].f
  if pd.notna(v):a.append(v);b.append(d.iloc[i+1].close/d.iloc[i].close-1)
 if len(a)>=8 and np.std(a)>0 and np.std(b)>0:out.append((dt,spearmanr(a,b).statistic,len(a)))
x=pd.DataFrame(out,columns=['date','ic','n']);print('dates',len(x),'assets',len(ds),'mean_n',x.n.mean(),'coverage',x.n.mean()/15)
print('daily',x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean())
for h in [5,10,20]:
 z=[]
 for dt in dates:
  a=[];b=[]
  for s,d in ds.items():
   i=d.index.get_loc(dt)
   if i+h>=len(d):continue
   if pd.notna(d.iloc[i].f):a.append(d.iloc[i].f);b.append(d.iloc[i+h].close/d.iloc[i].close-1)
  if len(a)>=8 and np.std(a)>0 and np.std(b)>0:z.append(spearmanr(a,b).statistic)
 z=pd.Series(z);print(h,len(z),z.mean(),z.mean()/z.std(),(z>0).mean())
print('recent',x.tail(250).ic.mean(),x.tail(250).ic.mean()/x.tail(250).ic.std())
