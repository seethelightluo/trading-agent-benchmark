import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
    try: d=get_stock_daily_data(s, days=3000)
    except Exception: d=None
    if d is not None and len(d): data[s]=d[['date','close']].set_index('date')
try: v=get_index_daily_data('VIX', days=3000)
except Exception: v=None
if v is not None: data['VIX']=v[['date','close']].set_index('date')
prices=pd.DataFrame({s:v.close for s,v in data.items()}).sort_index(); rets=prices.pct_change()
sig=-rets[U].where((rets.SPX<0)&(rets.NDX<0),0.0); fwd=rets[U].shift(-1)
ics=[]; rows=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(a)>=8 and a.iloc[:,0].nunique()>1 and a.iloc[:,1].nunique()>1:
  q=a.iloc[:,0].rank().corr(a.iloc[:,1].rank())
  if np.isfinite(q): ics.append(q); rows.append((dt,len(a),q))
ics=np.array(ics); print('assets',len(data),'dates',len(ics),'rows',sum(x[1] for x in rows),'avg_names',np.mean([x[1] for x in rows]))
print('daily_ic %.6f daily_icir %.6f hit %.4f activation %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),np.mean(ics>0),np.mean([(rets.SPX<0).loc[x[0]] and (rets.NDX<0).loc[x[0]] for x in rows])))
for h in [5,10]:
 f=rets[U].rolling(h).sum().shift(-h); z=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1 and a.iloc[:,1].nunique()>1:
   q=a.iloc[:,0].rank().corr(a.iloc[:,1].rank())
   if np.isfinite(q): z.append(q)
 z=np.array(z); print('%dd_ic %.6f icir %.6f dates %d'%(h,z.mean(),z.mean()/z.std(ddof=1),len(z)))
print('start',sig.index.min(),'end',sig.index.max())
