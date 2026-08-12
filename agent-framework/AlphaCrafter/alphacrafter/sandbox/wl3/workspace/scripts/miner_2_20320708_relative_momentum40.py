import os, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 df=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: df=fn(s,days=5000)
  except Exception: df=None
  if df is not None and len(df): break
 if df is not None and len(df):
  x=df[['date','close']].copy(); x.date=pd.to_datetime(x.date); frames[s]=x.drop_duplicates('date').set_index('date').close
px=pd.DataFrame(frames).sort_index().ffill(); ret=px.pct_change(40); fac=ret.sub(ret.mean(axis=1),axis=0).rolling(5,min_periods=5).mean().shift(1); fwd=px.shift(-10)/px-1
rows=[]; sig=[]
for d in fac.index:
 z=pd.DataFrame({'fac':fac.loc[d],'fwd':fwd.loc[d]}).dropna()
 if len(z)>=8:
  rows.append((d,z.fac.corr(z.fwd,method='spearman'),len(z)))
  sig += [(d,s,float(v)) for s,v in z.fac.items()]
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mean=ic.ic.mean(); sd=ic.ic.std(ddof=1); icir=mean/sd*np.sqrt(252)
r=pd.DataFrame(sig,columns=['date','symbol','factor']).pivot(index='date',columns='symbol',values='factor'); turnover=r.rank(axis=1,pct=True).diff().abs().mean().mean(); coverage=ic.n.sum()/(len(ic)*len(U))
print({'assets':len(frames),'dates':len(ic),'avgN':ic.n.mean(),'coverage':coverage,'IC':mean,'ICIR':icir,'hit':(ic.ic>0).mean(),'turnover':turnover})
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 q=ic.loc[a:b].ic
 if len(q): print(a+'-'+b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
q=ic.ic.tail(120); print('recent120',q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
r.stack().rename('factor').to_csv('scripts/miner_2_20320708_relative_momentum40_signal.csv'); ic.to_csv('scripts/miner_2_20320708_relative_momentum40_ic.csv')
