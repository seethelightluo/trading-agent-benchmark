import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2027-04-21'); D={}
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); x=x[x.date<=end].copy()
 x['r1']=x.close.pct_change(); x['r5']=x.close.pct_change(5); x['r20']=x.close.pct_change(20); x['f1']=x.close.shift(-1)/x.close-1; x['f5']=x.close.shift(-5)/x.close-1
 D[s]=x.set_index('date')
rows=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 a=[]
 for s,x in D.items():
  if d in x.index:
   q=x.loc[d]
   if np.isfinite([q.r1,q.r5,q.r20,q.f1,q.f5]).all():
    # continuous agreement: reversal strength multiplied by bounded trend agreement
    agreement=np.sign(q.r5*q.r20)*min(abs(q.r5),abs(q.r20))/(abs(q.r5)+abs(q.r20)+1e-12)
    a.append((s,-q.r1*agreement,q.f1,q.f5))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','sig','f1','f5']); rows.append((d,spearmanr(z.sig,z.f1).statistic,spearmanr(z.sig,z.f5).statistic,len(a)))
r=pd.DataFrame(rows,columns=['date','ic1','ic5','n']).set_index('date').dropna()
print('dates',len(r),'rows',int(r.n.sum()),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.sum()/(len(r)*15),4))
for c in ['ic1','ic5']:
 print(c,'IC',round(r[c].mean(),6),'ICIR',round(r[c].mean()/r[c].std(ddof=1),6),'hit',round((r[c]>0).mean(),4))
for y,g in r.groupby(r.index.year): print(y,'n',len(g),'ic1',round(g.ic1.mean(),5),'ir1',round(g.ic1.mean()/g.ic1.std(ddof=1),5))
