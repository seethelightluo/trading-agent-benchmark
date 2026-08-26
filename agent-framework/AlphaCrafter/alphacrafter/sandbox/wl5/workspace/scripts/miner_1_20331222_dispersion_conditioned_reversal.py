import json, numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2033-12-21')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date'],usecols=['date','close']).sort_values('date'); px[s]=d[d.date<=CUT].set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); r5=p.pct_change(5); v20=r.rolling(20,min_periods=15).std()*np.sqrt(252)
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); mu=disp.rolling(252,min_periods=100).mean(); sd=disp.rolling(252,min_periods=100).std(); regime=np.tanh((disp-mu)/(sd+1e-12))
f=(-r5/(v20+1e-8)).mul(1+0.625*regime,axis=0); y=p.shift(-10)/p-1
ics=[]; rows=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna(); z.columns=['f','y']
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
  c=z.f.corr(z.y,method='spearman')
  if np.isfinite(c): ics.append((dt,c)); ns.append(len(z))
 for s in f.columns:
  if pd.notna(f.loc[dt,s]): rows.append({'date':dt,'symbol':s,'factor_value':float(f.loc[dt,s]),'signal':float(f.loc[dt,s])})
i=pd.DataFrame(ics,columns=['date','ic']); a=i.ic.to_numpy(); rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean(); pd.DataFrame(rows).to_csv('scripts/miner_1_20331222_dispersion_conditioned_reversal_signal.csv',index=False)
print(json.dumps({'dates':len(i),'instruments':len(U),'start':str(i.date.min().date()),'end':str(i.date.max().date()),'mean_n':float(np.mean(ns)),'coverage':float(len(ns)*np.mean(ns)/(len(i)*len(U))),'IC':float(a.mean()),'ICIR':float(a.mean()/a.std(ddof=1)*np.sqrt(252)),'hit_ratio':float(np.mean(a>0)),'turnover':float(turn),'artifact_rows':len(rows)}))
for a1,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-12-21')]:
 q=i[(i.date>=a1)&(i.date<=b)]; print('REGIME',a1,b,len(q),float(q.ic.mean()) if len(q) else None,float(q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252)) if len(q)>1 else None)
for h in [5,10,20]:
 yy=p.shift(-h)/p-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('DECAY',h,len(vals),float(np.nanmean(vals)))
