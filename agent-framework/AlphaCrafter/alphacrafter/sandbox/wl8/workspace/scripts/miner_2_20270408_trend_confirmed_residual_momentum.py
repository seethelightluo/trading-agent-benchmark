import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-04-07'); D={}
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); x=x[x.date<=end].copy()
 x['r5']=x.close.pct_change(5); x['r20']=x.close.pct_change(20); x['fwd1']=x.close.shift(-1)/x.close-1
 D[s]=x.set_index('date')[['r5','r20','fwd1','close']]
rows=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 a=[]
 for s,x in D.items():
  if d in x.index and np.isfinite(x.loc[d,['r5','r20','fwd1']]).all(): a.append((s,x.loc[d,'r5'],x.loc[d,'r20'],x.loc[d,'fwd1']))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','r5','r20','fwd'])
  # residual short momentum, activated only when 20d trend agrees
  z['sig']=z.r5-z.r5.median()
  z.loc[np.sign(z.r5)!=np.sign(z.r20),'sig']*=0.0
  rows.append((d,spearmanr(z.sig,z.fwd).statistic,len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'rows',int(r.n.sum()),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.sum()/(len(r)*15),4))
print('daily',round(r.ic.mean(),6),round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4),'turnover_proxy',round((r.ic.diff().abs()>0).mean(),4))
for y,g in r.groupby(r.index.year): print(y,len(g),round(g.ic.mean(),6),round(g.ic.mean()/g.ic.std(ddof=1),6))
for h in [1,5,10]:
 vals=[]
 for d in sorted(set().union(*[set(x.index) for x in D.values()])):
  a=[]
  for s,x in D.items():
   if d in x.index:
    i=x.index.get_loc(d)
    if i+h<len(x) and np.isfinite(x.iloc[i][['r5','r20']]).all():
     sig=x.iloc[i].r5-x.groupby(level=0).r5.transform('median').iloc[i] if False else x.iloc[i].r5
     a.append((s,sig,x.iloc[i+h].close/x.iloc[i].close-1))
  if len(a)>=8:
   z=pd.DataFrame(a,columns=['s','sig','fwd']); z['sig']=z.sig-z.sig.median(); vals.append(spearmanr(z.sig,z.fwd).statistic)
 print('horizon',h,'dates',len(vals),'IC',round(np.nanmean(vals),6),'ICIR',round(np.nanmean(vals)/np.nanstd(vals,ddof=1),6))
