import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-09-09')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close
 px[s]=d.sort_index()
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); bm=R.median(axis=1)
F=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 up=R[s].where(bm>0).rolling(60,min_periods=45).mean(); dn=R[s].where(bm<0).rolling(60,min_periods=45).mean(); F[s]=up-dn
for h in [1,5,10]:
 vals=[]; Y=P.shift(-h)/P-1
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']); a.date=pd.to_datetime(a.date); a=a.set_index('date'); print('h',h,'dates',len(a),'avgN',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC %.5f ICIR %.5f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
 if h==1:
  for y in sorted(a.index.year.unique()):
   q=a[a.index.year==y]; print('regime',y,len(q),'IC %.5f ICIR %.5f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
rr=F.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rr)):
 z=pd.concat([rr.iloc[i-1],rr.iloc[i]],axis=1).dropna()
 if len(z)>=8: ts.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('rank_turnover',round(np.nanmean(ts),5))
