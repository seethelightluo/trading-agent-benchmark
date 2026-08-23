import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-03-24'); D={}
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); x=x[x.date<=end]
 x['sig']=-(x.close/x.open-1); x['fwd']=x.close.shift(-1)/x.close-1
 D[s]=x.set_index('date')[['sig','fwd','close']]
rows=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 a=[]
 for s,x in D.items():
  if d in x.index and np.isfinite(x.loc[d,'sig']) and np.isfinite(x.loc[d,'fwd']): a.append((s,x.loc[d,'sig'],x.loc[d,'fwd']))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','sig','fwd']); rows.append((d,spearmanr(z.sig,z.fwd).statistic,len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'rows',int(r.n.sum()),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
print('daily',r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for y,g in r.groupby(r.index.year): print(y,len(g),round(g.ic.mean(),5),round(g.ic.mean()/g.ic.std(ddof=1),5))
for h in [1,3,5,10]:
 rr=[]
 for d in sorted(set().union(*[set(x.index) for x in D.values()])):
  a=[]
  for s,x in D.items():
   if d in x.index:
    i=x.index.get_loc(d); sig=x.iloc[i].sig
    if i+h<len(x) and np.isfinite(sig): a.append((sig,x.iloc[i+h].close/x.iloc[i].close-1))
  if len(a)>=8: rr.append(spearmanr(np.array(a)[:,0],np.array(a)[:,1]).statistic)
 print('horizon',h,'dates',len(rr),'IC',round(np.nanmean(rr),5),'ICIR',round(np.nanmean(rr)/np.nanstd(rr,ddof=1),5))
