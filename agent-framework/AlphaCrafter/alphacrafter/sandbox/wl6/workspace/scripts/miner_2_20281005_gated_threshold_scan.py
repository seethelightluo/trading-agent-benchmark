import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
for gate in [.5,1,1.5,2]:
 a=[]
 for dt in dates:
  if dt>pd.Timestamp('2028-10-04'):continue
  v={};y={}
  for s,x in D.items():
   z=x.loc[:dt]
   if len(z)<25:continue
   r=np.log(z.iloc[-1]/z.iloc[-4]);q=np.log(z).diff().rolling(20).std().iloc[-1]
   if pd.isna(q) or q==0:continue
   v[s]=np.clip(-r/q,-3,3) if abs(r)/q>=gate else 0
   f=x[x.index>dt]
   if len(f):y[s]=f.iloc[0]/z.iloc[-1]-1
  c=list(set(v)&set(y));
  if len(c)>=8 and np.std([v[s] for s in c])>0:a.append(spearmanr([v[s] for s in c],[y[s] for s in c]).statistic)
 print(gate,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1),(np.array(a)>0).mean())
