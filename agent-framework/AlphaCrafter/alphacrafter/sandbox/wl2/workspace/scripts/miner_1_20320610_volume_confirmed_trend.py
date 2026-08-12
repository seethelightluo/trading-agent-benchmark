import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            d=fn(s,days=5000)
            if d is not None and len(d)>=100:
                d=d.copy(); d.date=pd.to_datetime(d.date); return d.sort_values('date').drop_duplicates('date').set_index('date')
        except Exception: pass
    return None
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); ret=px.pct_change()
# Volume-confirmed trend: medium trend scaled by volatility and strengthened by relative volume.
vol=ret.rolling(30).std().replace(0,np.nan)
trend=px.pct_change(20)/(vol*np.sqrt(20))
volume=pd.DataFrame({s:d.volume.astype(float) for s,d in D.items()}).reindex(px.index)
relvol=volume/volume.rolling(30).median().replace(0,np.nan)
# bounded confirmation avoids domination by crypto volume anomalies
confirm=(np.log(relvol.clip(.25,4))).clip(-1,1)
f=trend*(1+0.35*confirm)
rows=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],ret.shift(-1).loc[dt]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: rows.append((dt,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'dates',len(px),'IC_dates',len(o),'avg_n',round(o.n.mean(),2),'coverage',round(o.n.mean()/len(U),4))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic; print(a+'-'+b,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for h in [1,3,5,10]:
 rr=px.pct_change(h).shift(-h); vals=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,round(float(np.nanmean(vals)),6),len(vals))
print('recent120',round(o.tail(120).ic.mean(),6),round(o.tail(120).ic.mean()/o.tail(120).ic.std(),6),len(o.tail(120)))
f.to_csv('scripts/miner_1_20320610_volume_confirmed_trend_signal.csv',index_label='date')
