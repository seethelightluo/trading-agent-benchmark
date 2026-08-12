import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: d=fn(s,days=5000)
  except Exception: d=None
  if d is not None and len(d): break
 if d is not None and len(d):
  q=d[['date','close']].copy(); q.date=pd.to_datetime(q.date)
  F[s]=q.drop_duplicates('date').set_index('date').close
px=pd.DataFrame(F).sort_index().ffill(); ret=px.pct_change(); vol=ret.rolling(20).std()
# Trend persistence: risk-adjusted 20d return weighted by the fraction of positive daily returns.
persistence=(ret.gt(0).rolling(20).mean()-0.5)*2
fac=(px.pct_change(20)/vol*persistence).shift(1)
fwd=px.shift(-10)/px-1
rows=[]; sig=[]
for d in fac.index:
 z=pd.DataFrame({'fac':fac.loc[d],'fwd':fwd.loc[d]}).dropna()
 if len(z)>=8:
  rows.append((d,z.fac.corr(z.fwd,method='spearman'),len(z)))
  sig.extend((d,s,float(v)) for s,v in z.fac.items())
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
r=pd.DataFrame(sig,columns=['date','symbol','factor']).pivot(index='date',columns='symbol',values='factor')
m=ic.ic.mean(); sd=ic.ic.std(ddof=1)
print({'assets':len(F),'dates':len(ic),'avgN':ic.n.mean(),'coverage':ic.n.sum()/(len(ic)*len(U)),'IC':m,'ICIR':m/sd*np.sqrt(252),'hit':(ic.ic>0).mean(),'turnover':r.rank(axis=1,pct=True).diff().abs().mean().mean()})
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 q=ic.loc[a:b].ic
 if len(q)>1: print(a+'-'+b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
for n in [60,120,252]:
 q=ic.ic.tail(n); print('recent',n,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
pd.DataFrame(sig,columns=['date','symbol','factor']).to_csv('scripts/miner_2_20320819_trend_persistence_signal.csv',index=False)
ic.to_csv('scripts/miner_2_20320819_trend_persistence_ic.csv')
