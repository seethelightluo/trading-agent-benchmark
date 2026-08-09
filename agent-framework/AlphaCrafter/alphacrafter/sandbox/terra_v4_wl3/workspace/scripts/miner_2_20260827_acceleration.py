import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s, days=3000)
 if d is None or len(d)<100: return None
 d=d.copy(); d['date']=pd.to_datetime(d.date).dt.normalize(); d=d.drop_duplicates('date').set_index('date').sort_index()
 return d
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# acceleration: recent trend versus prior trend, scaled by recent realized risk
for mode in ['accel','accel_vol']:
 rows=[]
 for s,d in D.items():
  r=d['close'].pct_change()
  r5=d['close'].pct_change(5); r20=d['close'].pct_change(20)
  if mode=='accel': f=r5-r20/4
  else: f=(r5-r20/4)/(r.rolling(20).std()*np.sqrt(20))
  # shift feature one completed day; forward next close return
  x=pd.DataFrame({'f':f,'y':r.shift(-1)}).dropna()
  x['s']=s; rows.append(x.reset_index())
 X=pd.concat(rows,ignore_index=True)
 ics=[]; nms=[]
 for dt,g in X.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: ics.append(g.f.corr(g.y)); nms.append(len(g))
 a=np.array(ics); print(mode,'dates',len(a),'avg_names',np.mean(nms),'coverage',len(X)/sum(len(d) for d in D.values()),'IC %.5f ICIR %.5f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
 for h in [5,10]:
  rows=[]
  for s,d in D.items():
   r=d.close.pct_change(); f=(r.rolling(5).sum()-r.rolling(20).sum()/4) if mode=='accel' else (r.rolling(5).sum()-r.rolling(20).sum()/4)/(r.rolling(20).std()*np.sqrt(20))
   z=pd.DataFrame({'f':f,'y':d.close.pct_change(h).shift(-h)}).dropna(); z=z.reset_index(names='date'); rows.append(z)
  Z=pd.concat(rows)
  aa=[]
  for dt,g in Z.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1: aa.append(g.f.corr(g.y))
  aa=np.array(aa); print(' h',h,'dates',len(aa),'IC %.5f ICIR %.5f'%(np.nanmean(aa),np.nanmean(aa)/np.nanstd(aa,ddof=1)))
