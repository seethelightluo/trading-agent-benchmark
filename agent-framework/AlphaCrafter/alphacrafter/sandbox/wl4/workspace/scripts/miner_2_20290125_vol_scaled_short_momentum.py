import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
U={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4];d=pd.read_csv(f);d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date');d['ret']=d.close.pct_change();U[s]=d
dates=sorted(set.intersection(*[set(x.index) for x in U.values()])); rows=[]
for dt in dates:
 v={};fw={}
 for s,d in U.items():
  if dt not in d.index:continue
  x=d.loc[:dt].tail(21)
  if len(x)<21:continue
  r=x.close.iloc[-1]/x.close.iloc[-6]-1; vol=x.ret.iloc[-20:].std()
  v[s]=r/(vol*np.sqrt(20)+.01)
  q=d.loc[d.index>dt,'close'].head(10)
  if len(q)>=10:fw[s]=q.iloc[-1]/d.loc[dt,'close']-1
 c=set(v)&set(fw)
 if len(c)>=8:rows.append((dt,spearmanr([v[s] for s in c],[fw[s] for s in c]).statistic,len(c)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for k,z in [('all',r),('r500',r.tail(500)),('r250',r.tail(250))]:print(k,len(z),z.n.mean(),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),z.n.mean()/15)
