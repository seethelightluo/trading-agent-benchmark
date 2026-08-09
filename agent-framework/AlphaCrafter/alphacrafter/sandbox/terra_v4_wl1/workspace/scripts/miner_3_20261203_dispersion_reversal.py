import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-03')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); P=P[P.index<=cut]
R=P.pct_change(); r3=R.rolling(3,min_periods=3).sum(); csdisp=R.apply(lambda x:x.std(),axis=1).rolling(3,min_periods=3).mean(); q=csdisp.rolling(60,min_periods=30).rank(pct=True); amp=(0.5+q).clip(0.5,1.5); F=-r3.mul(amp,axis=0)
print('sample',P.index.min(),P.index.max(),'rows',len(P))
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.y,method='spearman'))); ns.append(len(z))
 a=pd.DataFrame(vals,columns=['date','ic']).set_index('date').ic
 print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 if h==1:
  for yr,g in a.groupby(a.index.year): print('year',yr,'n',len(g),'IC',g.mean())
print('coverage',F.notna().sum().sum()/F.size,'rankturn',F.rank(axis=1,pct=True).diff().abs().mean().mean())
for start in ['2024-01-01','2025-01-01','2026-01-01']:
 Y=P.pct_change().shift(-1); z=[]
 for dt in P.loc[start:].index:
  x=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(x)>=8:z.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
 z=pd.Series(z); print('recent',start,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
