import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allx={}
for s in UNIV:
 try: d=get_stock_daily_data(s,days=4000)
 except Exception: d=None
 if d is not None and len(d): allx[s]=d[['date','close']].drop_duplicates('date').set_index('date').close.rename(s)
for s in ['DXY','VIX']:
 try: d=get_index_daily_data(s,days=4000)
 except Exception: d=None
 if d is not None and len(d): allx[s]=d[['date','close']].drop_duplicates('date').set_index('date').close.rename(s)
px=pd.concat(allx,axis=1).sort_index().loc['2020-01-01':]
r=np.log(px).diff(); out=[]
for t in px.index:
 hist=r.loc[:t].tail(61)
 if len(hist)<55: continue
 vals={}
 for s in UNIV:
  if s not in r: continue
  h=hist[[s,'DXY','VIX']].dropna()
  if len(h)<45: continue
  X=np.column_stack([np.ones(len(h)),h[['DXY','VIX']].values]); y=h[s].values
  b=np.linalg.lstsq(X,y,rcond=None)[0]
  rr20=r.loc[:t,s].tail(20).sum()-b[1]*r.loc[:t,'DXY'].tail(20).sum()-b[2]*r.loc[:t,'VIX'].tail(20).sum()
  vol=h[s].tail(40).std(); vals[s]=rr20/(vol*np.sqrt(20)+1e-12)
 if len(vals)>=8: out.append((t,vals))
rows=[{'date':t,'symbol':s,'signal':x} for t,v in out for s,x in v.items()]; sig=pd.DataFrame(rows)
fwd=np.log(px).shift(-1)-np.log(px); recs=[]
for t,g in sig.groupby('date'):
 y=fwd.loc[t].reindex(g.symbol).dropna(); z=g.set_index('symbol').signal.reindex(y.index).dropna(); y=y.reindex(z.index)
 if len(z)>=8: recs.append((t,z.rank().corr(y.rank()),len(z)))
ic=pd.Series({t:x for t,x,n in recs}).sort_index()
print('endpoint',px.index.max().date(),'dates',len(ic),'avgN',np.mean([n for t,x,n in recs]),'coverage',len(sig)/(len(px)*len(UNIV)))
for label,q in [('full',ic),('2028+',ic[ic.index>='2028-01-01']),('2029+',ic[ic.index>='2029-01-01']),('2030',ic[ic.index>='2030-01-01'])]: print(label,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/(q.std(ddof=1)+1e-12),'hit',np.mean(q>0))
path='scripts/miner_2_20300627_macro_residual_momentum_signal.csv';sig.to_csv(path,index=False);print('artifact',path)
