import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float).replace(0,np.nan)
 except: pass
p=pd.DataFrame(D).sort_index(); r=np.log(p).diff()
# Volatility-shock reversal: recent 5d move is faded more aggressively after an idiosyncratic volatility expansion.
rv5=r.rolling(5,min_periods=4).std(); rv30=r.rolling(30,min_periods=20).std()
shock=(rv5/(rv30+1e-12)-1).clip(-2,3)
ret5=np.log(p/p.shift(5)); f=(-ret5*(1+0.7*shock)).rolling(2,min_periods=2).mean()
q=np.log(p.shift(-10)/p); rows=[]; sig=[]
for d in f.index:
 z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): rows.append((d,c,len(z)))
  for s in z.index: sig.append((d,s,float(f.loc[d,s])))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]).sort_index(); ns=np.array([x[2] for x in rows])
print('dates',len(i),'avgN',ns.mean(),'coverage',ns.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(ddof=1),(i>0).mean()))
for k in [365,750,1260]:
 z=i.tail(k); print('recent',k,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
for a,b in [('2024','2027'),('2028','2030'),('2031','2033')]:
 z=i.loc[a:b]; print('regime',a+'-'+b,len(z),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 qq=np.log(p.shift(-h)/p); aa=[]
 for d in f.index:
  z=pd.concat([f.loc[d],qq.loc[d]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'%.6f'%np.nanmean(aa))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20330331_volshock_reversal_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i.values,'n':ns}).to_csv('scripts/miner_2_20330331_volshock_reversal_ic.csv',index=False)
