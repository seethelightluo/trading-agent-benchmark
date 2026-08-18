import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 d=get_stock_daily_data(s,4200); return pd.Series(d.close.values,index=pd.to_datetime(d.date))
P=pd.DataFrame({s:L(s) for s in A}); R=np.log(P).diff();
# trend quality: 60d return divided by realized vol, with path efficiency confirmation
m=R.rolling(60).sum(); v=R.rolling(60).std()*np.sqrt(60); eff=m.abs()/(R.abs().rolling(60).sum()+1e-12)
f=m/(v+1e-12)*eff
F=f; ic=[]; ns=[]
for t in F.index:
 y=np.log(P.shift(-10)/P).loc[t]; x=F.loc[t]; ok=x.notna()&y.notna()
 if ok.sum()>=8: ic.append(x[ok].corr(y[ok])); ns.append(ok.sum())
ic=pd.Series(ic).dropna(); print('dates',len(ic),'avg_n',np.mean(ns),'coverage',F.notna().mean().mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turnover',F.rank(pct=True).diff().abs().mean(axis=1).mean())
for w in [120,252,504]:
 q=ic.tail(w);print('recent',w,q.mean(),q.mean()/q.std())
for h in [1,5,10,20]:
 y=np.log(P.shift(-h)/P); z=[]
 for t in F.index:
  ok=F.loc[t].notna()&y.loc[t].notna()
  if ok.sum()>=8:z.append(F.loc[t,ok].corr(y.loc[t,ok]))
 q=pd.Series(z).dropna();print('decay',h,q.mean(),q.mean()/q.std())
print('period',ic.index.min(),ic.index.max())
