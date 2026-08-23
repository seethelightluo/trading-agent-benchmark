import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float)
 except: pass
p=pd.DataFrame(D).sort_index();r=np.log(p).diff();v=r.rolling(20,min_periods=15).std()*np.sqrt(20)
base=-np.log(p/p.shift(10))/(v+1e-8); disp=r.rolling(20,min_periods=15).std().mean(axis=1)
# Smooth cross-sectional reversal, weighted toward high recent cross-asset dispersion
scale=(disp/disp.rolling(252,min_periods=60).median()).clip(.5,2.0)
f=base.mul(scale,axis=0).rolling(3,min_periods=2).mean(); rows=[]
for d in f.index:
 q=np.log(p.shift(-10)/p);z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c):rows.append((d,c,len(z)))
i=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]).sort_index();n=np.array([x[2] for x in rows])
print('dates',len(i),'avgN',n.mean(),'coverage',n.mean()/15);print('IC %.6f ICIR %.6f hit %.4f'%(i.mean(),i.mean()/i.std(),(i>0).mean()))
for k in [365,750,1260]:
 z=i.tail(k);print('recent',k,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std()))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032')]:
 z=i.loc[a:b];print('regime',a+'-'+b,len(z),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 q=np.log(p.shift(-h)/p);a=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a))
pd.DataFrame([(d,s,f.loc[d,s]) for d in f.index for s in f.columns if pd.notna(f.loc[d,s])],columns=['date','symbol','signal']).to_csv('scripts/miner_1_20330106_dispersion_scaled_reversal10_signal.csv',index=False)
pd.DataFrame({'date':i.index,'ic':i.values,'n':n}).to_csv('scripts/miner_1_20330106_dispersion_scaled_reversal10_ic.csv',index=False)