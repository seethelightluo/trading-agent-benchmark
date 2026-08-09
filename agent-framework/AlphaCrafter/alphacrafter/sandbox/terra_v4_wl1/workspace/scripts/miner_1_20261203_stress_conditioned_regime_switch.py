import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-03'); base='../persistent/stock_data'; macro='../persistent/index_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
P=pd.DataFrame(px).sort_index(); P=P[P.index<=cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill()
sp=P['SPX'].ffill(); stress=((sp.pct_change(20)<0)&(v.pct_change(5)>0))|(v>v.rolling(60,min_periods=40).median())
r5=-P.pct_change(5); r20=P.pct_change(20)
f=r5.where(pd.DataFrame(np.repeat(stress.values[:,None],len(U),axis=1),index=P.index,columns=P.columns),r20)
Y={h:P.shift(-h).div(P)-1 for h in [1,5,10]}
for h in [1,5,10]:
 rows=[]
 for dt in P.index:
  q=pd.concat([f.loc[dt].rename('f'),Y[h].loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((dt,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1:
  print('stress_fraction',round(stress.reindex(a.index).mean(),4))
  for yr,g in ic.groupby(ic.index.year): print('year',yr,'IC',round(g.mean(),5),'ICIR',round(g.mean()/g.std(ddof=1),4),'n',len(g))
print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),4),'rankturn',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
