import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d['close'] for s,d in D.items()}).loc[:cut]
V=pd.DataFrame({s:d['volume'] if 'volume' in d else np.nan for s,d in D.items()}).reindex(P.index)
R=P.pct_change(); mom=R.rolling(20,min_periods=15).sum(); vs=(V/(V.rolling(60,min_periods=30).median())).replace([np.inf,-np.inf],np.nan)
# interpretable signed trend, emphasizing moves confirmed by unusual activity
F=mom*np.log(vs.clip(lower=.25,upper=4.0))
print('rows',len(P),'range',P.index.min(),P.index.max(),'assets',P.notna().sum().to_dict())
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 a=pd.DataFrame(vals,columns=['date','ic']).set_index('date').ic
 print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==1:
  for yr,g in a.groupby(a.index.year): print('year',yr,'n',len(g),'IC',round(g.mean(),6),'ICIR',round(g.mean()/g.std(ddof=1),4))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
# independent proxy correlations to existing likely factors
base={'rev5':-R.rolling(5,min_periods=5).sum(),'mom20':mom,'clv':(D[U[0]]['close']*0)}
for k,x in base.items():
 aa=F.stack().corr(x.reindex_like(F).stack(),method='spearman');print('corr',k,round(aa,4))
for start in ['2020-01-01','2022-01-01','2024-01-01','2025-01-01','2026-01-01']:
 a=[];Y=P.pct_change().shift(-1)
 for dt in P.loc[start:].index:
  z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a);print('recent',start,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
