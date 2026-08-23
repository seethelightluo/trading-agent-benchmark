import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

end=pd.Timestamp('2026-07-15')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,3000) if s!='VIX' else get_index_daily_data(s,3000)
 if d is None:return pd.DataFrame()
 d=d.copy();d['date']=pd.to_datetime(d['date']);return d[d.date<=end].set_index('date').sort_index()
a={s:load(s) for s in watch}; v=load('VIX')
# trailing conditional return: mean asset return on VIX-positive days, normalized by unconditional vol
vr=v['close'].pct_change(); dates=sorted(set(v.index).intersection(*[set(x.index) for x in a.values()]))
rows=[]
for t in dates:
 past=[x for x in dates if x<t][-60:]
 if len(past)<45:continue
 vals={}; fwd={}
 for s,d in a.items():
  r=d['close'].pct_change()
  q=r.reindex(past); z=vr.reindex(past)
  sel=q[z>0]
  if len(sel)>=8: vals[s]=sel.mean()/(q.std()+1e-12)
  if t in d.index: fwd[s]=d['close'].pct_change().shift(-1).get(t,np.nan)
 valid=[s for s in watch if s in vals and np.isfinite(fwd.get(s,np.nan))]
 if len(valid)>=8: rows.append((t, np.corrcoef([vals[s] for s in valid],[fwd[s] for s in valid])[0,1], vals, fwd))
ics=pd.Series({x[0]:x[1] for x in rows}).dropna();
print('dates',len(ics),'avg_names',np.mean([len(x[2]) for x in rows]),'coverage',len(rows)*len(watch) and sum(len(x[2]) for x in rows)/(len(rows)*len(watch)))
print('IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/(ics.std(ddof=1)+1e-12), (ics>0).mean()))
for h in [5,10]:
 hs=[]
 for t,_,vals,_ in rows:
  ff={s:(a[s]['close'].shift(-h)/a[s]['close']-1).get(t,np.nan) for s in vals}
  ss=[s for s in vals if np.isfinite(ff[s])]
  if len(ss)>=8:hs.append(np.corrcoef([vals[s] for s in ss],[ff[s] for s in ss])[0,1])
 print('horizon',h,'n',len(hs),'IC %.6f ICIR %.6f'%(np.mean(hs),np.mean(hs)/(np.std(hs,ddof=1)+1e-12)))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=ics[(ics.index>=lo)&(ics.index<=hi)];print('regime',lo,hi,len(z),'mean',z.mean(),'ICIR',z.mean()/(z.std(ddof=1)+1e-12))
