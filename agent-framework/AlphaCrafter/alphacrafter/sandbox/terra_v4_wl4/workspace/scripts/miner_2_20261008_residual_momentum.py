import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); return d.close.astype(float).loc[:'2026-10-07']
P=pd.DataFrame({s:load(s) for s in U}); R=P.pct_change(fill_method=None); m=R['SPX']; beta=pd.DataFrame(index=R.index,columns=U,dtype=float)
for s in U: beta[s]=R[s].rolling(60,min_periods=45).cov(m)/m.rolling(60,min_periods=45).var()
res=pd.DataFrame({s:R[s]-beta[s]*m for s in U}); factor=res.rolling(20,min_periods=20).sum()
for h in [1,5,10]:
 vals=[]; ns=[]; ds=[]
 for i in range(len(R)-h):
  z=pd.concat([factor.iloc[i].rename('x'),R[U].iloc[i+1:i+1+h].sum().rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.x.corr(z.y,method='spearman')); ns.append(len(z)); ds.append(R.index[i])
 q=pd.Series(vals,index=ds); print('h',h,'dates',len(q),'avg_n',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==1:
  for yr,g in q.groupby(q.index.year): print('regime',yr,'n',len(g),'ic %.5f icir %.5f'%(g.mean(),g.mean()/g.std()))
print('coverage',len(q)/(len(R)-1),'rank turnover',factor.rank(axis=1).diff().abs().stack().mean()/len(U))
factor.to_csv('/tmp/residual_momentum_signal.csv')
