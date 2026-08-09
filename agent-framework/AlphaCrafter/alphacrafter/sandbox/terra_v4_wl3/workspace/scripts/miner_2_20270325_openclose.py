import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);return d.set_index('date')
d={a:L(a) for a in A}; op=pd.DataFrame({a:x.open for a,x in d.items()}); cl=pd.DataFrame({a:x.close for a,x in d.items()}); f=(-(cl/op-1)).rank(axis=1,pct=True); y=cl.pct_change().shift(-1)
for h in [1,5,10]:
 yy=cl.pct_change(h).shift(-h);q=[]; ns=[]
 for t in f.index:
  z=pd.concat([f.loc[t],yy.loc[t]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 q=np.array(q);print(h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.diff().abs().stack().mean())
f.reset_index().to_csv('scripts/miner_2_20270325_openclose_signal.csv',index=False)
