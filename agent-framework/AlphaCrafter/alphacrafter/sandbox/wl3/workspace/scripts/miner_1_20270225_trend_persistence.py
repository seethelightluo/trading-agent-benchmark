import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=2200)
    if d is not None and len(d):
        d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').close
        frames[s]=d
px=pd.DataFrame(frames).sort_index()
r=px.pct_change()
# trend persistence: signed breadth of daily moves over 30 sessions, scaled by realized risk;
# avoids relying on one endpoint and rewards persistent advances
breadth=(r>0).rolling(30,min_periods=20).mean()-0.5
ret30=px.pct_change(30)
vol20=r.rolling(20,min_periods=15).std()
f=breadth + 0.20*ret30/(vol20*np.sqrt(30))
# forward one-day return, date-aligned and strict cutoff
cut=pd.Timestamp('2027-02-24')
f=f.loc[:cut]; fr=r.shift(-1).loc[:cut]
rows=[]; daily=[]
for dt in f.index:
    x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        daily.append((dt,ic,len(z)))
        for sym in z.index: rows.append((dt,sym,float(x[sym])))
d=pd.DataFrame(daily,columns=['date','ic','n'])
print('instruments',len(frames),'dates',len(d),'avg_n',d.n.mean(),'coverage',len(rows)/(len(f.index)*len(U)))
print('IC %.6f ICIR %.6f hit %.4f' %(d.ic.mean(),d.ic.mean()/d.ic.std(ddof=1), (d.ic>0).mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-02-24')]:
 q=d[(d.date>=a)&(d.date<=b)]; print(a,'n',len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
for h in [3,5,10]:
 fy=px.pct_change(h).shift(-h).loc[:cut]; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1),len(vals))
# rank turnover
rank=f.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna(); print('turnover',turn.mean())
# artifact
out=pd.DataFrame(rows,columns=['date','symbol','signal']); out.to_csv('scripts/miner_1_20270225_trend_persistence_signal.csv',index=False)
