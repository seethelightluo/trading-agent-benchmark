import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().loc[:'2033-05-11']
r=p.pct_change(); mom=p/p.shift(40)-1; vol=r.rolling(20).std()*np.sqrt(252); f=-(mom/vol)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],p.pct_change(10).shift(-10).loc[d]],axis=1).dropna()
 if len(q)>=8: rows.append((d,len(q),q.iloc[:,0].corr(q.iloc[:,1])))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('data',p.index.min(),p.index.max(),'dates',len(a),'avgN',a.n.mean(),'coverage',f.notna().sum(axis=1).mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean()))
for k in [1,5,10,20]:
 rr=[]; fr=p.pct_change(k).shift(-k)
 for d in f.index:
  q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8: rr.append(q.iloc[:,0].corr(q.iloc[:,1]))
 print('decay',k,np.nanmean(rr),len(rr))
for label,sub in [('recent365',a[a.index>=a.index.max()-pd.Timedelta(days=365)]),('from2030',a[a.index>='2030-01-01']),('pre2030',a[a.index<'2030-01-01'])]: print(label,len(sub),'IC %.6f ICIR %.6f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20330512_riskadj_momentum_40d_signal.csv',index=False); a.reset_index().to_csv('scripts/miner_2_20330512_riskadj_momentum_40d_ic.csv',index=False)
