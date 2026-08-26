import numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-06-14')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.loc[:end] for s in U}).sort_index()
R=px.pct_change(); neg=R.clip(upper=0); down=np.sqrt((neg**2).rolling(20,min_periods=15).mean())
fac=(px/px.shift(20)-1)/(down+0.002); rows=[]
for j,dt in enumerate(px.index):
 for h in [1,5,10,20]:
  if j+h>=len(px): continue
  y=(px.iloc[j+h]/px.iloc[j]-1).rename('y'); q=pd.concat([fac.iloc[j].rename('f'),y],axis=1).dropna()
  if len(q)>=8: rows.append((dt,h,len(q),spearmanr(q.f,q.y).statistic))
A=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('period',px.index.min().date(),end.date(),'instruments',len(U),'observations',len(A),'coverage',round(fac.notna().mean().mean(),4),'avgN',round(A.n.mean(),2))
for h,g in A.groupby('h'):
 ic=g.ic.mean(); ir=ic/g.ic.std(ddof=1)
 print('horizon',h,'dates',len(g),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((g.ic>0).mean(),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
 for name,cut in [('online','2026-07-16'),('recent252','2027-06-15'),('ytd','2028-01-01')]:
  z=g[g.date>=pd.Timestamp(cut)]
  if len(z): print(name,len(z),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6),round((z.ic>0).mean(),4))
