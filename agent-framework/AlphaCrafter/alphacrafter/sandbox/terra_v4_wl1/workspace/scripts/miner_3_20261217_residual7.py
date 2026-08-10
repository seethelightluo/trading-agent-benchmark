import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17'); P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float); P[s]=d[d.index<=END]
P=pd.DataFrame(P).sort_index(); R=P.pct_change(); m=R.mean(axis=1); w=60
mr=m.rolling(w,min_periods=40).mean(); vr=(m*m).rolling(w,min_periods=40).mean()-mr*mr
beta=R.mul(m,axis=0).rolling(w,min_periods=40).mean().sub(R.rolling(w,min_periods=40).mean().mul(mr,axis=0)).div(vr.replace(0,np.nan),axis=0)
res=R-beta.mul(m,axis=0); resid=(1+res).rolling(7,min_periods=7).apply(np.prod,raw=True)-1; rv=res.rolling(20,min_periods=15).std()*np.sqrt(20); F=-resid.div(rv.replace(0,np.nan))
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((d,z.f.corr(z.y,method='spearman'),len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=a.ic
 print('h',h,'dates',len(x),'avgN',round(a.n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 if h==1:
  for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026-12-17','25-26')]:
   q=x[(x.index>=lo)&(x.index<=hi)]; print(n,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20261219_residual7_signal.csv',index=False)
