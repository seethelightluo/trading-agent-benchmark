import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
r=P.pct_change(); b=r['SPX']
# Beta-neutral relative momentum: residual cumulative return after removing each asset's 60d SPX beta exposure.
rollcov=r.rolling(60,min_periods=40).cov(b); var=b.rolling(60,min_periods=40).var()
beta=rollcov.div(var,axis=0)
raw=P.pct_change(20); f=raw.sub(beta.mul(P['SPX'].pct_change(20),axis=0))
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y,method='spearman'),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'period',P.index.min(),P.index.max())
# pooled correlation to plain 20d momentum on overlapping signal values
m=raw.stack(); z=f.stack(); print('corr_plain',m.corr(z))
