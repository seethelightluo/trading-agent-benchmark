import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
from scipy.stats import spearmanr
acct=get_account_dict(); allowed=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
universe=acct.get('watch_list') or allowed; universe=[x for x in universe if x in allowed]
frames={}
for s in universe:
    try: d=get_index_daily_data(s, days=5000)
    except Exception: d=None
    if d is None:
        try: d=get_stock_daily_data(s, days=5000)
        except Exception: d=None
    if d is not None and len(d)>0:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').drop_duplicates('date'); frames[s]=d.set_index('date')
print('loaded',len(frames),sorted(frames))
all_dates=sorted(set.union(*[set(x.index) for x in frames.values()]))
rows=[]
for dt in all_dates:
 for s,d in frames.items():
  hist=d.loc[d.index<=dt]
  if len(hist)<25 or not {'high','low','close'}.issubset(hist.columns): continue
  h=hist.tail(20); rng=np.log(h['high'].clip(lower=1e-12)/h['low'].clip(lower=1e-12)); f=-np.sqrt(np.mean(rng.to_numpy()**2)/(4*np.log(2)))
  fut=d.loc[d.index>dt,'close']
  if len(fut)>=20 and hist['close'].iloc[-1]>0: rows.append((dt,s,float(f),fut))
def evaluate(horizon):
 vals=[]
 for dt,s,f,fut in rows:
  if len(fut)>=horizon: vals.append((dt,s,f,float(fut.iloc[horizon-1]/frames[s].loc[frames[s].index<=dt,'close'].iloc[-1]-1)))
 z=pd.DataFrame(vals,columns=['date','symbol','factor','fwd']); ics=[]
 for _,g in z.groupby('date'):
  if len(g)>=8:
   q=spearmanr(g.factor,g.fwd).statistic
   if np.isfinite(q): ics.append(q)
 a=np.array(ics); print('horizon',horizon,'dates',len(a),'avg_n',z.groupby('date').size().mean(),'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for h in [1,5,10,20]: evaluate(h)
print('coverage %.4f'%(len(set((x[0],x[1]) for x in rows))/max(1,len(set(x[0] for x in rows))/len(frames))))
