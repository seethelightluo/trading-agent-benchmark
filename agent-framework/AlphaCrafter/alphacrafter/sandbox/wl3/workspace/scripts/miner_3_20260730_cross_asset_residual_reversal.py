import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
R=pd.concat({s:D[s].close.pct_change() for s in U},axis=1).sort_index()
for w in [10,20,40]:
 X=R.sub(R.median(axis=1),axis=0); V=X.rolling(w,min_periods=max(5,w//2)).std(); F=-X/V
 rows=[]
 for s in U:
  q=pd.DataFrame({'f':F[s], 'y':R[s].shift(-1)});q['date']=q.index;rows.append(q.reset_index(drop=True))
 A=pd.concat(rows,ignore_index=True).dropna(); obs=[]
 for dt,g in A.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1: obs.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
 O=pd.DataFrame(obs,columns=['date','ic','n']); x=O.ic
 turn=F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2
 print('window',w,'dates',len(x),'avgN',O.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',len(A)/ (R.shape[0]*15),'turn',turn)
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=O[(O.date.dt.year>=lo)&(O.date.dt.year<=hi)].ic;print(' regime',lo,hi,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'N',len(z))
 for h in [5,10]:
  Y=[]
  for s in U:
   q=pd.DataFrame({'f':F[s], 'y':D[s].close.pct_change(h).shift(-h)});q['date']=q.index;Y.append(q.reset_index(drop=True))
  B=pd.concat(Y,ignore_index=True).dropna(); z=[]
  for dt,g in B.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
  print(' decay',h,np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1))
