import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:cut]
r=P.pct_change(); v=r.rolling(20,min_periods=15).std(); vprev=v.shift(10)
# Volatility compression: reward declining recent volatility, an interpretable state/mean-reversion signal.
f=-(v/vprev-1)
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y,method='spearman'),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,len(g),round(g.mean(),4),round(g.mean()/g.std(ddof=1),4))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'period',P.index.min(),P.index.max())
