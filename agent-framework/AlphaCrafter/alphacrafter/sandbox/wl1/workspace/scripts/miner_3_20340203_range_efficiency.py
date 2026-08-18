import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    d=get_stock_daily_data(s,5000)
    if d is None or len(d)==0: d=get_index_daily_data(s,5000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); P[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill()
# range-efficiency momentum: directional displacement divided by total path length,
# scaled by recent volatility. Observable at date t, signal is lagged one session.
r=px.pct_change()
ret20=px.pct_change(20)
path20=r.abs().rolling(20).sum()
vol20=r.rolling(20).std()
# interpretable: persistent trend gets high absolute efficiency, direction retained
f=(ret20/(path20+1e-12)) / (vol20+1e-12)
# Winsorize cross-section each date to limit crypto outliers
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0).shift(1)
rows=[]
for h in [1,5,10,20]:
    fr=px.pct_change(h).shift(-h)
    ics=[]; dates=[]; ns=[]
    for dt in f.index:
        a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(a)>=8:
            c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
            if np.isfinite(c): ics.append(c); dates.append(dt); ns.append(len(a))
    z=np.array(ics); ic=z.mean(); sd=z.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd>0 else np.nan
    print('H',h,'dates',len(z),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(ic,icir,np.mean(z>0)))
    if h==10:
        sig=f.rank(axis=1,pct=True); turnover=sig.diff().abs().mean(axis=1).dropna().mean()
        print('coverage %.4f turnover %.4f'%(f.notna().sum(axis=1).mean()/len(U),turnover))
        out=f.loc[dates].copy(); out.insert(0,'date',out.index); out.to_csv('scripts/miner_3_20340203_range_efficiency_signal.csv',index=False)
# regime blocks
fr=px.pct_change(10).shift(-10)
for start,end in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 z=[]
 for dt in f.index:
  if not(start<=str(dt.year)<=end): continue
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 z=np.array(z); print('REG',start,end,'n',len(z),'IC %.6f ICIR %.6f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1)*np.sqrt(252) if len(z)>1 else np.nan))
print('range',px.index.min(),px.index.max(),'assets',len(P))
