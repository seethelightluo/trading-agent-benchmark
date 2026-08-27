import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 path='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(path):
  x=pd.read_csv(path);x['date']=pd.to_datetime(x['date']);D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();mom=p/p.shift(60)-1;vol=r.rolling(60).std()*np.sqrt(252);breadth=(r.rolling(20).sum()>0).mean(axis=1);f=mom/(vol+1e-8)*(0.5+breadth)
def ev(h):
 fr=p.shift(-h)/p-1;a=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()));ns.append(len(z));ds.append(dt)
 return pd.Series(a,index=ds),ns
q,ns=ev(10);print('candidate=macro_conditioned_momentum assets=%d dates=%d avgN=%.2f minN=%d coverage=%.2f'%(len(p.columns),len(q),np.mean(ns),min(ns),f.notna().sum(axis=1).mean()/len(U)));print('H10 IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std()*np.sqrt(252),(q>0).mean()))
for h in [5,20,40,60]:
 a,_=ev(h);print('H%d IC %.6f ICIR %.6f'%(h,a.mean(),a.mean()/a.std()*np.sqrt(252)))
for a,b in [(2020,2023),(2024,2026),(2027,2029),(2030,2032),(2033,2035)]:
 z=q[(q.index.year>=a)&(q.index.year<=b)];print('REG',a,b,len(z),z.mean(),z.mean()/z.std()*np.sqrt(252))
print('turn10 %.6f'%((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(10)).abs().mean(axis=1).mean()));f.index.name='date';f.to_csv('scripts/miner_1_20350913_macro_conditioned_momentum_signal.csv')
