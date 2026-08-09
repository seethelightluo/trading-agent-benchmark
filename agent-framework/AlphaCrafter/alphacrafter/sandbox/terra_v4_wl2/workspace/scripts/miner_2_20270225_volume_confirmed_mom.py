import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for a in assets:
 d=get_stock_daily_data(a,days=4000)
 if d is None or len(d)<100: continue
 d=d.copy(); d['date']=pd.to_datetime(d['date']); c=d['close'].astype(float); v=d['volume'].astype(float).replace(0,np.nan)
 # momentum whose strength is confirmed by recent-vs-long volume participation
 mom=c.pct_change(20); vr=v.rolling(5,min_periods=5).mean()/v.rolling(60,min_periods=30).mean()-1
 sig=(mom*vr).replace([np.inf,-np.inf],np.nan)
 fr=c.shift(-1)/c-1
 for dt,x,y in zip(d.date,sig,fr): rows.append((dt,a,x,y))
z=pd.DataFrame(rows,columns=['date','asset','signal','fwd']).dropna()
ics=[]; nms=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1:
  ics.append(g.signal.corr(g.fwd,method='spearman')); nms.append(len(g))
ic=np.array(ics); print('dates',len(ic),'avg_names',np.mean(nms),'IC %.8f ICIR %.8f hit %.4f coverage %.4f turnover_pending'% (np.nanmean(ic),np.nanmean(ic)/np.nanstd(ic,ddof=1),np.mean(ic>0),len(z)/(len(rows))))
for name,cut in [('2020-22',('2020','2022')),('2023-24',('2023','2024')),('2025-26',('2025','2026')),('2027',('2027','2027'))]:
 q=z[(z.date.dt.strftime('%Y')>=cut[0])&(z.date.dt.strftime('%Y')<=cut[1])]; a=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1:a.append(g.signal.corr(g.fwd,method='spearman'))
 a=np.array(a); print(name,len(a), 'IC %.8f ICIR %.8f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)) if len(a)>1 else 'NA')
for h in [3,5,10]:
 # recompute forward h using original not retained; approximate using signal dataframe unavailable; skip
 pass
