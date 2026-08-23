import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-06-02')
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date'); v=v[v.index<=END]; vr=v.close.pct_change(); vm=vr.rolling(20,min_periods=20).median(); D={}
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date'); x=x[x.index<=END]; x['r']=x.close.pct_change(); x['f']=x.close.shift(-1)/x.close-1; D[s]=x
rows=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 a=[]
 for s,x in D.items():
  if d in x.index:
   q=x.loc[d]; vv=vr.get(d,np.nan); mm=vm.get(d,np.nan)
   if np.isfinite([q.r,q.f,vv,mm]).all(): a.append((-q.r if vv>mm else 0.,q.f))
 if len(a)>=8:
  ic=spearmanr(np.array(a)[:,0],np.array(a)[:,1]).statistic
  if np.isfinite(ic): rows.append((d,ic,len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('revalidation through',END.date(),'dates',len(r),'rows',int(r.n.sum()),'avg_names',round(r.n.mean(),2),'coverage',round(r.n.sum()/(len(r)*15),4)); print('IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4));
for y,g in r.groupby(r.index.year): print(y,len(g),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),6))
for h in [20,60,252]:
 g=r.tail(h); print('recent',h,'dates',len(g),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),6))
