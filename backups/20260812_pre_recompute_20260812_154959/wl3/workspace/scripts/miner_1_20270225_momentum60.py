import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=get_stock_daily_data(s,days=2200)
 if d is not None and len(d): p[s]=d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close
px=pd.DataFrame(p).sort_index(); r=px.pct_change(); cut=pd.Timestamp('2027-02-24')
f=px.pct_change(60)/(r.rolling(60,min_periods=40).std()*np.sqrt(60)); f=f.loc[:cut]
res=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: res.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
d=pd.DataFrame(res,columns=['date','ic','n'])
print('instruments',len(p),'dates',len(d),'avg_n',d.n.mean(),'coverage',f.notna().sum().sum()/(len(f)*len(U)))
print('IC %.6f ICIR %.6f hit %.4f'%(d.ic.mean(),d.ic.mean()/d.ic.std(ddof=1),(d.ic>0).mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-02-24')]:
 q=d[(d.date>=a)&(d.date<=b)]; print(a,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 y=px.pct_change(h).shift(-h).loc[:cut]; zlist=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:zlist.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(zlist),np.nanmean(zlist)/np.nanstd(zlist,ddof=1),len(zlist))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
# write artifact
out=f.stack().rename('signal').rename_axis(['date','symbol']).reset_index();out.to_csv('scripts/miner_1_20270225_momentum60_signal.csv',index=False)
