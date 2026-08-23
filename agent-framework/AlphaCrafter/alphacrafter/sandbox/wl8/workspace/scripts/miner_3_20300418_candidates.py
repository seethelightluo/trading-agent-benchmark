import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
  D[s]=x['close'].astype(float)
 except Exception as e: print('missing',s,e)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
r20=np.log(p/p.shift(20)); r60=np.log(p/p.shift(60)); v20=r.rolling(20).std().shift(1); v60=r.rolling(60).std().shift(1)
f=((r20/v20)*(1+0.5*np.sign(r20)*np.sign(r60))).clip(-8,8)
f2=((r60/v60)*(1+0.5*(v60/(v20+1e-12)-1).clip(-1,1))).clip(-8,8)
def evalfac(z,name):
 rows=[]
 for d in z.index:
  fut=np.log(p.shift(-10)/p).loc[d]; a=z.loc[d]; ok=a.notna()&fut.notna()
  if ok.sum()>=8: rows.append((d,a[ok].rank().corr(fut[ok].rank()),ok.sum()))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
 print(name,'dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',ir,'hit',(q.ic>0).mean(),'turn',z.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
 for k in [180,360]: print(name,'recent',k,q.tail(k).ic.mean())
 for h in [5,20]:
  rr=[]
  for d in z.index:
   fut=np.log(p.shift(-h)/p).loc[d]; a=z.loc[d]; ok=a.notna()&fut.notna()
   if ok.sum()>=8: rr.append(a[ok].rank().corr(fut[ok].rank()))
  print(name,'decay',h,np.nanmean(rr))
evalfac(f,'agreement_momentum');evalfac(f2,'compression_momentum')
