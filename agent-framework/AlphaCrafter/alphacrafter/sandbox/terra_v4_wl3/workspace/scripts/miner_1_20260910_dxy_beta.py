import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent'; P={}
for s in U:
 d=pd.read_csv(f'{root}/stock_data/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); P[s]=d.close
P=pd.DataFrame(P).sort_index().loc[:'2026-07-15']; R=P.pct_change(fill_method=None)
dxy=pd.read_csv(f'{root}/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.reindex(P.index).ffill(); D=dxy.pct_change(); vd=D.rolling(40,min_periods=30).var()
F=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U: F[s]=-R[s].rolling(40,min_periods=30).cov(D).div(vd)
rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],R.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((str(dt),z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']); a['date']=pd.to_datetime(a.date); a=a.set_index('date'); ic=a.ic
print('dates',len(a),'mean_n %.2f coverage %.4f'%(a.n.mean(),F.notna().sum().sum()/F.size))
print('IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean()))
for h in [5,10]:
 yy=R.shift(-h).rolling(h).sum().shift(-(h-1)); q=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna();print('%dd IC %.6f ICIR %.6f n %d'%(h,q.mean(),q.mean()/q.std(ddof=1),len(q)))
print(a.assign(year=a.index.year).groupby('year').ic.agg(['mean','count']))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean()); F.to_csv('scripts/miner_1_20260910_dxy_beta_signal.csv')
