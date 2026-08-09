import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# low-vol trend: negative 20d realized vol, with volatility-of-vol stabilization
frames={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is None: d=get_index_daily_data(s,2500)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d['r']=d.close.pct_change(); frames[s]=d.set_index('date')
# date aligned factor known at t and forward returns t+1,t+5,t+10
all_dates=sorted(set().union(*[set(x.index) for x in frames.values()]))
rows={h:[] for h in [1,5,10]}; daily=[]
for dt in all_dates:
 vals=[]; fwd={h:[] for h in [1,5,10]}
 for s,d in frames.items():
  if dt not in d.index: continue
  ix=d.index.get_loc(dt)
  if ix<40: continue
  r=d.r.iloc[:ix+1]
  if r.iloc[-20:].isna().any() or ix+10>=len(d): continue
  vol=r.iloc[-20:].std()
  vv=r.iloc[-40:].rolling(10).std().iloc[-1]
  if not np.isfinite(vol) or not np.isfinite(vv): continue
  # prefer stable low realized volatility, mildly reward recent vol decline
  fac=-np.log(vol+1e-8) - 0.25*np.log(vv+1e-8)
  vals.append((s,fac))
  for h in [1,5,10]: fwd[h].append(d.close.iloc[ix+h]/d.close.iloc[ix]-1)
 if len(vals)>=8:
  x=np.array([z[1] for z in vals]);
  for h in [1,5,10]:
   y=np.array(fwd[h]); rows[h].append(np.corrcoef(x,y)[0,1])
  daily.append((dt,len(vals)))
def stats(a):
 a=np.array(a); a=a[np.isfinite(a)]; return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)) if len(a)>1 and a.std(ddof=1)>0 else 0,float((a>0).mean())
print('dates',len(daily),'avg_names',np.mean([n for _,n in daily]))
for h in [1,5,10]: print(h,stats(rows[h]))
for y in range(2020,2027):
 z=[rows[1][i] for i,(dt,n) in enumerate(daily) if pd.Timestamp(dt).year==y]
 print('regime',y,stats(z))
print('coverage',sum(n for _,n in daily)/(len(daily)*15))
