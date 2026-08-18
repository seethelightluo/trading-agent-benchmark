import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
CUTOFF=pd.Timestamp('2029-01-24')
U={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date'); d['ret']=d.close.pct_change(); U[s]=d
# factor is negative trailing downside deviation (lower downside risk ranks higher)
rows=[]
for dt in sorted(set.intersection(*[set(d.index) for d in U.values()])):
 if dt>CUTOFF: continue
 vals={}; fw={}
 for s,d in U.items():
  if dt not in d.index: continue
  x=d.loc[:dt].tail(21)
  f=x.ret.iloc[-20:]; down=np.sqrt(np.mean(np.minimum(f,0)**2))
  if len(x)==21 and np.isfinite(down): vals[s]=-down
  fut=d.loc[(d.index>dt)&(d.index<=CUTOFF),'close'].head(10)
  if len(fut)>=10: fw[s]=fut.iloc[-1]/d.loc[dt,'close']-1
 c=set(vals)&set(fw)
 if len(c)>=8: rows.append((dt,spearmanr([vals[s] for s in c],[fw[s] for s in c]).statistic,len(c)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name,z in [('all',r),('recent500',r.tail(500)),('recent250',r.tail(250))]:
 print(name,'dates',len(z),'avgN',z.n.mean(),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4),'coverage',round(z.n.mean()/15,4))
print('period',r.index.min(),r.index.max())
