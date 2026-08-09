import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-03')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for s in U}).sort_index()
P=P.loc[:cut].ffill(limit=3); R=P.pct_change(); common=R.mean(axis=1); res=R.sub(common,axis=0); F=-res.rolling(5,min_periods=5).sum()
print('sample',P.index.min().date(),P.index.max().date(),'rows',len(P),'assets',len(U),'dates_with_8',int((R.notna().sum(axis=1)>=8).sum()))
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); rows=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.f.corr(z.y,method='spearman'),len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1: print('years',[(int(y),round(g.mean(),5),len(g)) for y,g in ic.groupby(ic.index.year)])
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for start in ['2024-01-01','2025-01-01','2026-01-01']:
 z=[]; Y=P.pct_change().shift(-1)
 for dt in P.loc[start:].index:
  q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 s=pd.Series(z); print('recent',start,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6))
print('signal_artifact ready')
