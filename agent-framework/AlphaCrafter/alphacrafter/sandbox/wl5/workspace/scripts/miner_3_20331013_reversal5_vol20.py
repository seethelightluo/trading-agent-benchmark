import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d)>=100: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); f=(-R.rolling(5,min_periods=5).sum()/R.rolling(20,min_periods=15).std()).replace([np.inf,-np.inf],np.nan); fr=R.shift(-10).rolling(10,min_periods=10).sum()
a=[];ds=[];ns=[];rk=[];rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c):a.append(c);ds.append(dt);ns.append(len(z));rk.append(f.loc[dt].rank(pct=True))
 for s in f.columns:
  if pd.notna(f.loc[dt,s]):rows.append((dt,s,float(f.loc[dt,s])))
a=np.array(a); S=pd.DataFrame(rk,index=ds); pd.DataFrame(rows,columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20331013_reversal5_vol20_signal.csv',index=False); ir=a.mean()/a.std(ddof=1)*np.sqrt(252)
print({'dates':len(a),'start':str(ds[0].date()),'end':str(ds[-1].date()),'mean_n':round(np.mean(ns),3),'coverage':round(np.mean(ns)/15,6),'IC':round(a.mean(),6),'ICIR':round(ir,6),'hit':round(np.mean(a>0),6),'turnover':round(S.diff().abs().mean().mean(),6)})
for x,y in [('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-10-01')]:
 z=a[(np.array(ds)>=pd.Timestamp(x))&(np.array(ds)<=pd.Timestamp(y))];print(x,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1)*np.sqrt(252),6))
