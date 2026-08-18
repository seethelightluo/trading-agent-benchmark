import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 q=pd.read_csv('../persistent/stock_data/'+s+'.csv'); q.date=pd.to_datetime(q.date)
 D[s]=q.sort_values('date').set_index('date').close.astype(float)
p=pd.concat(D,axis=1).sort_index().ffill(); lr=np.log(p).diff(); ew=lr.mean(axis=1)
res=lr.rolling(30,min_periods=22).sum().sub(ew.rolling(30,min_periods=22).sum(),axis=0)
vol=lr.rolling(30,min_periods=22).std()*np.sqrt(30)
bench60=ew.rolling(60,min_periods=45).sum()
f=(res/(vol+1e-8)) * (1+0.5*np.tanh(bench60/0.08)).values[:,None]
f=f.shift(1); y=np.log(p).shift(-20)-np.log(p); rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8:
  v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(v): rows.append((d,v,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2020-01-02':'2033-10-12']
print('candidate residual_momentum_benchmark_confirmed_30d')
print('dates',len(x),'avgN',round(x.n.mean(),3),'coverage',round(x.n.mean()/15,4))
print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=x.loc[a:b]
 if len(q): print(a,b,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(),6))
print('rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[x.index].mean(),6))
for h in [5,10,20]:
 yy=np.log(p).shift(-h)-np.log(p); rr=[]
 for d in f.index:
  z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): rr.append(v)
 print('horizon',h,'IC',round(np.mean(rr),6),'ICIR',round(np.mean(rr)/np.std(rr),6),'dates',len(rr))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20331014_residual_benchmark_confirmed_signal.csv',index=False)
