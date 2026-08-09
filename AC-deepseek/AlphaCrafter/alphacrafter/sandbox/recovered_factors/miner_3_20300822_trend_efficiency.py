import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame(px).sort_index().astype(float); r=p.pct_change()
# Lagged trend efficiency: directional 20-observation return divided by total path variation.
# All observations are shifted one day so the signal is available before the forward window.
ret20=p.shift(1)/p.shift(21)-1
path=r.abs().rolling(20,min_periods=15).sum().shift(1)
sig=(ret20/path).replace([np.inf,-np.inf],np.nan)
print('candidate=lagged_trend_efficiency_20obs')
print('dates',len(p),'instruments',len(p.columns),'coverage',round(sig.notna().sum().sum()/sig.size,6),'meanN',round(sig.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 vals=[];ns=[]; f=p.shift(-h)/p-1
 for d in p.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1
 for label,mask in [('2020-23',p.index<'2024-01-01'),('2024-27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('2028+',p.index>='2028-01-01'),('latest120',p.index>=p.index[-120])]:
  a=[]
  for d in p.index[mask]:
   z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.asarray(a); print('regime',h,label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
# component orthogonality diagnostics
for name,c in [('ret20',ret20),('inv_path',1/path),('ret5',p.shift(1)/p.shift(6)-1)]:
 z=pd.concat([sig.stack().rename('s'),c.stack().rename('c')],axis=1).dropna();print('corr',name,round(spearmanr(z.s,z.c).statistic,6),'cells',len(z))
