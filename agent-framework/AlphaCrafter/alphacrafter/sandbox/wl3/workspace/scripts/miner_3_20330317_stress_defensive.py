import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=U[:8]
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); logp=np.log(P); r=logp.diff()
m20=logp-logp.shift(20); m5=logp-logp.shift(5); eq=m20[E].mean(axis=1); eq5=m5[E].mean(axis=1)
rel=m20.sub(eq,axis=0)
# Stress confirmation: sustained equity weakness and broad negative breadth; lagged and smoothed.
breadth=(m20[E]<0).mean(axis=1)
conds={'stress_breadth':(eq<0)&(breadth>=0.625),'stress_confirm':(eq<0)&(eq5<0)&(breadth>=0.5),'stress_hysteresis':((eq<0)&(breadth>=0.5)).rolling(5,min_periods=1).max().shift(1).fillna(0).astype(bool)}
fr=logp.shift(-10)-logp
for name,c in conds.items():
 f=(rel*c.astype(float).values[:,None]).rolling(3,min_periods=3).mean().shift(1)
 rows=[]
 for dt in f.index:
  a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
 print(name,'dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
 for n in [120,252,756]:
  x=q.tail(n); print(' recent',n,round(x.mean()/x.std(ddof=1),5))
 f.to_csv('scripts/miner_3_20330317_'+name+'_signal.csv'); z.to_csv('scripts/miner_3_20330317_'+name+'_ic.csv')
