import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
L=10
xs={}
for s in U:
 d=get_stock_daily_data(s, days=3000)
 if d is not None and len(d):
  z=d[['date','close']].copy(); z['date']=pd.to_datetime(z.date); z=z.drop_duplicates('date').set_index('date').close
  xs[s]=z.pct_change(L)
p=pd.DataFrame(xs).sort_index()
# lagged, leave-one-out peer median
rows=[]
for dt in p.index:
 vals=p.loc[dt]
 if vals.notna().sum()<8: continue
 for s in U:
  if s not in p or pd.isna(vals.get(s)): continue
  peer=vals.drop(labels=s).dropna()
  if len(peer)<7: continue
  # factor is prior 10d peer performance, forward return constructed below
  rows.append((dt,s,float(peer.median())))
f=pd.DataFrame(rows,columns=['date','symbol','factor']).set_index(['date','symbol'])
# forward one-day return, and 5/10 day decay
close=pd.DataFrame({s:get_stock_daily_data(s,days=3000)[['date','close']].assign(date=lambda x:pd.to_datetime(x.date)).drop_duplicates('date').set_index('date').close for s in xs}).sort_index()
rets={h:close.pct_change(h).shift(-h) for h in [1,5,10]}
for h,r in rets.items():
 rr=r.stack().rename('fwd'); q=f.join(rr,how='inner').dropna()
 by=q.groupby(level=0).apply(lambda x:x.factor.corr(x.fwd) if len(x)>=8 and x.factor.nunique()>1 and x.fwd.nunique()>1 else np.nan).dropna()
 ic=by.mean(); sd=by.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
 print('horizon',h,'dates',len(by),'obs',len(q),'avgN',len(q)/len(by),'IC %.6f ICIR %.6f hit %.4f'%(ic,icir,(by>0).mean()))
# turnover rank changes
rank=f.groupby(level=0).factor.rank(pct=True)
print('turnover',rank.groupby(level=1).diff().abs().groupby(level=0).mean().mean(),'coverage',len(f)/((len(p.index))*15))
print('period',p.index.min(),p.index.max())
