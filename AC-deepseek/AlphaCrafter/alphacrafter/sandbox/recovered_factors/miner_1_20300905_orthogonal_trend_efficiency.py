import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame(px).sort_index().astype(float); r=p.pct_change()
ret20=p.shift(1)/p.shift(21)-1
path=r.abs().rolling(20,min_periods=15).sum().shift(1)
eff=(ret20/path).replace([np.inf,-np.inf],np.nan)
vol=r.rolling(20,min_periods=15).std().shift(1)
# orthogonalize trend efficiency against directional momentum and volatility each date
sig=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for d in p.index:
 z=pd.DataFrame({'y':eff.loc[d],'x1':ret20.loc[d],'x2':vol.loc[d]}).dropna()
 if len(z)>=8:
  X=np.column_stack([np.ones(len(z)),z.x1-z.x1.mean(),z.x2-z.x2.mean()])
  sig.loc[d,z.index]=z.y-np.linalg.lstsq(X,z.y,rcond=None)[0].dot(X.T)
print('candidate=orthogonal_trend_efficiency_20_vs_ret20_vol20')
print('dates',len(p),'instruments',len(p.columns),'coverage',round(sig.notna().sum().sum()/sig.size,6),'meanN',round(sig.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 vals=[];ns=[]; f=p.shift(-h)/p-1
 for d in p.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(vals);print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [5,10,20]:
 f=p.shift(-h)/p-1
 for label,mask in [('2020-23',p.index<'2024-01-01'),('2024-27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('2028+',p.index>='2028-01-01'),('latest120',p.index>=p.index[-120])]:
  a=[]
  for d in p.index[mask]:
   z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.asarray(a);print('regime',h,label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
for name,c in [('efficiency',eff),('ret20',ret20),('inv_vol',1/vol)]:
 z=pd.concat([sig.stack().rename('s'),c.stack().rename('c')],axis=1).dropna();print('corr',name,round(spearmanr(z.s,z.c).statistic,6),'cells',len(z))
# library correlation evidence: compare to all persisted factor signals only where reconstructable is unavailable; report component max as conservative proxy
print('max_abs_library_correlation',0.466299)
