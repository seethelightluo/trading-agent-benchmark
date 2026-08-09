import numpy as np,pandas as pd
from pathlib import Path
root=Path('../persistent'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-10-07')
def load(s): return pd.read_csv(root/'stock_data'/(s+'.csv'),parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
pd0=pd.concat({s:load(s) for s in U},axis=1).sort_index().loc[:C].dropna(); r=pd0.pct_change(); b=r.mean(axis=1); bm=b.rolling(60,min_periods=45).mean(); bv=((b-bm)**2).rolling(60,min_periods=45).mean(); f=pd.DataFrame(index=r.index,columns=U,dtype=float)
for s in U:
 x=r[s]; xm=x.rolling(60,min_periods=45).mean(); cov=((x-xm)*(b-bm)).rolling(60,min_periods=45).mean(); beta=cov/bv; f[s]=(x-beta*b).rolling(20,min_periods=15).sum()
f=f.replace([np.inf,-np.inf],np.nan)
for h in [1,5,10]:
 y=pd0.shift(-h).div(pd0)-1; q=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 a=pd.DataFrame(q,columns=['date','ic','n']).set_index('date'); print('h',h,'dates',len(a),'avgN',a.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
 if h==1:
  print('coverage %.5f turnover %.5f'%(a.n.sum()/(len(a)*15),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
  for y0,y1 in [('2020','2022'),('2023','2024'),('2025','2026')]:
   bb=a.loc[y0:y1]; print('regime',y0,y1,len(bb),'IC %.6f ICIR %.6f'%(bb.ic.mean(),bb.ic.mean()/bb.ic.std(ddof=1)))
out=[{'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(f.loc[d,s])} for d in f.index for s in U if pd.notna(f.loc[d,s])]; pd.DataFrame(out).to_csv('scripts/miner_3_20261008_residual_momentum_signal.csv',index=False); print('cutoff',C.date(),'rows',len(out),'common_dates',len(pd0))
