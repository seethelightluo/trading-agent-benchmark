import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float) for s in U}
p=pd.DataFrame(D).sort_index(); r=np.log(p).diff(); vol=r.rolling(20,min_periods=12).std()*np.sqrt(20)
f=(-(np.log(p/p.shift(10)))/(vol+1e-8)).rolling(3,min_periods=2).mean()
disp=r.sub(r.mean(axis=1),axis=0).std(axis=1); gate=disp.shift(1)>disp.shift(1).rolling(252,min_periods=126).quantile(.80); f=f.where(gate)
rows=[]; sig=[]
for d in f.index:
 z=pd.concat([f.loc[d],np.log(p.shift(-10)/p).loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): rows.append((d,c,len(z)))
  for s in z.index: sig.append((d,s,f.loc[d,s]))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]).sort_index(); n=np.array([x[2] for x in rows])
print('dates',len(i),'avgN %.3f coverage %.4f'%(n.mean(),n.mean()/15)); print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for k in [365,750,1260]:
 q=i.tail(k); print('recent',k,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
for a,b in [('2026','2028'),('2029','2032')]:
 q=i.loc[a:b]; print('regime',a,len(q),'IC %.6f ICIR %.6f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('turnover %.6f'%f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 q=np.log(p.shift(-h)/p); a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'IC %.6f'%np.nanmean(a))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20330203_extreme_disp_reversal_signal.csv',index=False);pd.DataFrame({'date':i.index,'ic':i.values,'n':n}).to_csv('scripts/miner_1_20330203_extreme_disp_reversal_ic.csv',index=False)
