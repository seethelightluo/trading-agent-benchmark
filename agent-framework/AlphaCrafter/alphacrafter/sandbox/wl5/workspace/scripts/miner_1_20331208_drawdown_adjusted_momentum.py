import json
import numpy as np
import pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2033-12-07')
px={}
for s in U:
    d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date')
    px[s]=d[d.date<=CUT].set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().ffill()
r20=p.pct_change(20)
# Risk-adjusted momentum: trailing 20-session return divided by the
# magnitude of the worst close-to-close drawdown over the preceding 60 sessions.
rollmax=p.rolling(60,min_periods=40).max()
dd=p/rollmax-1
risk=-dd.rolling(60,min_periods=40).min()
f=r20/(risk+0.02)
y=p.shift(-10)/p-1
rows=[]; ics=[]; ns=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna(); z.columns=['f','y']
    if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
        ic=z.f.corr(z.y,method='spearman')
        if np.isfinite(ic):
            ics.append((dt,ic)); ns.append(len(z))
            for s in z.index: rows.append({'date':dt,'symbol':s,'signal':float(z.loc[s,'f']),'forward_return_10d':float(z.loc[s,'y'])})
i=pd.DataFrame(ics,columns=['date','ic']); meanic=i.ic.mean(); icir=meanic/i.ic.std(ddof=1)*np.sqrt(252)
rank=f.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).dropna().mean()
print(json.dumps({'dates':len(i),'instruments':len(U),'start':str(i.date.min().date()),'end':str(i.date.max().date()),'mean_n':float(np.mean(ns)),'coverage':float(len(rows)/(len(i)*len(U))),'IC':float(meanic),'ICIR':float(icir),'hit_ratio':float((i.ic>0).mean()),'turnover':float(turnover)},default=str))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-12-07')]:
 q=i[(i.date>=a)&(i.date<=b)]
 if len(q)>1: print('REGIME',a,b,len(q),float(q.ic.mean()),float(q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252)))
for h in [5,10,20]:
 yy=p.shift(-h)/p-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('DECAY',h,len(vals),float(np.nanmean(vals)))
out=pd.DataFrame(rows); out.to_csv('scripts/miner_1_20331208_drawdown_adjusted_momentum_signal.csv',index=False)
print('artifact_rows',len(out))
