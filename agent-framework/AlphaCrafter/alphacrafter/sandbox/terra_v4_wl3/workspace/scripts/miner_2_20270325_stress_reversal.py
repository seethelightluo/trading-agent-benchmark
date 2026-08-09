import pandas as pd, numpy as np, glob, os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p):
 d=pd.read_csv(p); d['date']=pd.to_datetime(d.date); return d.set_index('date').sort_index()
px={a:load('../persistent/stock_data/'+a+'.csv') for a in assets}
cl=pd.DataFrame({a:d.close for a,d in px.items()}); ret=cl.pct_change()
v=load('../persistent/index_data/VIX.csv').close.reindex(cl.index).ffill()
# stress-scaled 3-day reversal, only information through t
stress=(v/v.rolling(60,min_periods=30).median()).clip(0.5,3.0)
f=(-ret.rolling(3).sum()).mul(stress,axis=0)
# avoid cross-sectional scale effects; rank is interpretable and robust
f=f.rank(axis=1,pct=True)
fr=ret.shift(-1)
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(r),'avgN',r.n.mean(),'coverage',f.notna().sum().sum()/f.size)
print('IC %.8f ICIR %.8f hit %.6f turnover %.6f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean(), f.diff().abs().stack().mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-12-31')]:
 q=r.loc[lo:hi,'ic']; print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10]:
 yy=cl.pct_change(h).shift(-h); rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'dates',len(rr),'IC',np.nanmean(rr),'ICIR',np.nanmean(rr)/np.nanstd(rr,ddof=1))
r.reset_index().to_csv('scripts/miner_2_20270325_stress_reversal_signal.csv',index=False)
