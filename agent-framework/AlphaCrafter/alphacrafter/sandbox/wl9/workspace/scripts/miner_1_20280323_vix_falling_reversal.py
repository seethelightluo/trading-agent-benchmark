import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for root in ['../persistent/stock_data/','../persistent/index_data/']:
  f=root+s+'.csv'
  if os.path.exists(f): return pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close.astype(float)
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}; P=pd.DataFrame(D).sort_index().ffill(); R=np.log(P).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.reindex(P.index).ffill(); state=(v.pct_change(5)<0)
F=-R.rolling(3,min_periods=3).sum().where(state,0); rows=[]
for dt in F.index:
 for h in [1,5,10]:
  z=pd.concat([F.loc[dt],np.log(P.shift(-h).loc[dt]/P.loc[dt])],axis=1).dropna()
  if len(z)>=8 and state.loc[dt]: rows.append((dt,h,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=r[r.h==h]; print('h',h,'dates',len(q),'n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4));
 for label,sub in [('online',q[q.date>=pd.Timestamp('2026-07-16')]),('recent',q[q.date>=pd.Timestamp('2027-03-23')])]: print(label,len(sub),round(sub.ic.mean(),6) if len(sub) else None,round(sub.ic.mean()/sub.ic.std(ddof=1),6) if len(sub)>1 else None)
print('assets',len(D),'state_frac',round(state.mean(),3),'coverage',round(F.notna().sum(axis=1).mean()/len(D),3),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
