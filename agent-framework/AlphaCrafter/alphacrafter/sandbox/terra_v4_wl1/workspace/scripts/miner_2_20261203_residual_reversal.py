import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-12-03');b='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index();P=P[P.index<=cut];R=P.pct_change(); m=R['SPX']; beta=R.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0); resid=R.sub(beta.mul(m,axis=0)); f=-resid.rolling(5,min_periods=5).sum(); raw=-R.rolling(5).sum()
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');ic=a.ic
 print(h,len(ic),round(a.n.mean(),2),round(ic.mean(),6),round(ic.mean()/ic.std(),6),round((ic>0).mean(),4))
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turn',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'corrraw',f.stack().corr(raw.stack()))
print([(y,round(g.mean(),4),len(g)) for y,g in ic.groupby(ic.index.year)])
