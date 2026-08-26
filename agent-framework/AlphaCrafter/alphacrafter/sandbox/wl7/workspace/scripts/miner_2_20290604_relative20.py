import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=None
    try: d=get_index_daily_data(s,2600)
    except Exception: pass
    if d is None or len(d)<150:
        try: d=get_stock_daily_data(s,2600)
        except Exception: pass
    if d is not None and len(d)>100: frames[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(frames).sort_index().ffill(); r=p.pct_change()
# Relative strength: asset's trailing 20d return minus contemporaneous cross-sectional median,
# lagged one day to prevent look-ahead. Forward return is 10 trading days.
raw=p.shift(1)/p.shift(21)-1
f=raw.sub(raw.median(axis=1),axis=0)
ics=[]; rows=[]
for i in range(22,len(p)-10):
    z=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); ics.append(ic); rows.append([p.index[i],len(z),ic])
ics=np.asarray(ics); dates=pd.to_datetime([x[0] for x in rows])
turn=np.nanmean(np.abs(f.diff()).sum(axis=1)/f.notna().sum(axis=1))
print('candidate relative 20d momentum vs cross-sectional median')
print('assets',len(frames),'dates',len(ics),'avg_n',round(np.mean([x[1] for x in rows]),3),'coverage',round(np.mean([x[1] for x in rows])/15,4))
print('IC',round(np.nanmean(ics),6),'ICIR',round(np.nanmean(ics)/np.nanstd(ics,ddof=1),6),'hit',round(np.mean(ics>0),4),'turnover',round(turn,4))
for label,mask in [('2020-24',(dates>='2020-01-01')&(dates<'2025-01-01')),('2025-26',(dates>='2025-01-01')&(dates<'2027-01-01')),('2027-28',(dates>='2027-01-01')&(dates<'2029-01-01')),('recent',(dates>='2028-09-01'))]:
 q=ics[mask]; print(label,len(q),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6) if len(q)>1 else np.nan)
pd.DataFrame(rows,columns=['date','n','ic']).to_csv('scripts/miner_2_20290604_relative20_signal.csv',index=False)
