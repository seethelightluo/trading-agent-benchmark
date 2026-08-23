import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-05-05'); D={}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date'); v=v[v.date<=end].set_index('date'); vp=v['close'].reindex(v.index).ffill(); vm=vp.rolling(20,min_periods=20).median()
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); x=x[x.date<=end].copy(); x['r1']=x.close.pct_change(); x['f1']=x.close.shift(-1)/x.close-1; x=x.set_index('date'); x['vix']=vp.reindex(x.index); x['vixmed']=vm.reindex(x.index); D[s]=x
rows=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 a=[]
 for s,x in D.items():
  if d in x.index:
   q=x.loc[d]
   if np.isfinite([q.r1,q.f1,q.vix,q.vixmed]).all():
    # contrarian reversal only in elevated VIX regime, zero otherwise
    sig=-q.r1 if q.vix>q.vixmed else 0.0
    a.append((s,sig,q.f1))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','sig','f1']); ic=spearmanr(z.sig,z.f1).statistic
  if np.isfinite(ic): rows.append((d,ic,len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'rows',int(r.n.sum()),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.sum()/(len(r)*15),4))
print('IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4))
for y,g in r.groupby(r.index.year): print(y,'dates',len(g),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),6))
print('recent',r.tail(252).ic.mean(),r.tail(252).ic.mean()/r.tail(252).ic.std(ddof=1))
