import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

# Beta-neutral residual momentum: recent asset return stripped of rolling cross-asset factor exposure.
acct=get_account_dict(); uni=acct.get('watch_list',[])
if not uni: uni=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in uni:
 d=get_stock_daily_data(s,days=1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=1500)
 if d is not None: px[s]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); rets=prices.pct_change()
# common factor = daily cross-sectional mean, avoiding observation-only assets
mkt=rets.mean(axis=1)
# signal at t uses returns through t; validation pairs signal t with return t+1
rows=[]
for t in range(65,len(prices)-1):
 date=prices.index[t]
 vals={}
 for s in prices.columns:
  r=rets[s].iloc[t-19:t+1]; q=mkt.iloc[t-19:t+1]
  z=pd.concat([r,q],axis=1).dropna()
  if len(z)<15: continue
  vx=np.var(z.iloc[:,1],ddof=1)
  beta=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/vx if vx>1e-12 else 1
  resid=(rets[s].iloc[t-4:t+1].sum()-beta*mkt.iloc[t-4:t+1].sum())
  vol=rets[s].iloc[t-19:t+1].std()
  vals[s]= -resid/(vol*np.sqrt(20)) if vol>1e-8 else np.nan
 f=pd.Series(vals).replace([np.inf,-np.inf],np.nan); f=f.dropna()
 fwd=rets.iloc[t+1].reindex(f.index)
 z=pd.concat([f,fwd],axis=1).dropna()
 if len(z)>=8: rows.append((date, z.iloc[:,0].corr(z.iloc[:,1]), len(z), f))
ics=np.array([x[1] for x in rows]); ns=np.array([x[2] for x in rows])
print('factor beta-neutral 5d residual reversal; dates',len(rows),'avgN',ns.mean(),'coverage',ns.mean()/len(prices.columns))
print('IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),np.mean(ics>0)))
for label,cut in [('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 a=ics[[x[0]>=cut for x in rows]]; print(label,'dates',len(a),'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/a.std(ddof=1)) if len(a)>2 else 'NA')
# turnover of daily cross-sectional ranks
turn=[]; prev=None
for _,_,_,f in rows:
 rank=f.rank(pct=True)
 if prev is not None:
  turn.append(np.mean(np.abs(rank.reindex(prev.index).fillna(.5)-prev.reindex(rank.index).fillna(.5))))
 prev=rank
print('rank turnover',np.mean(turn))
for h in [3,5,10]:
 hh=[]
 for t in range(65,len(prices)-h):
  # recompute use stored nearest row date, then forward cumulative
  pass
 # use rows and price lookup
 for date,ic,n,f in rows:
  if date not in prices.index: continue
  j=prices.index.get_loc(date)
  if j+h>=len(prices): continue
  z=pd.concat([f, (prices.iloc[j+h]/prices.iloc[j]-1).reindex(f.index)],axis=1).dropna()
  if len(z)>=8: hh.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'IC %.6f ICIR %.6f dates %d'%(np.mean(hh),np.mean(hh)/np.std(hh,ddof=1),len(hh)))
