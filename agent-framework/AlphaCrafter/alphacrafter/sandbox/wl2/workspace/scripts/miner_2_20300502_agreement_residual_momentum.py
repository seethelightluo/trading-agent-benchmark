import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None:d=get_index_daily_data(s,4000)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d');F[s]=d.set_index('date')
c=pd.DataFrame({s:d.close for s,d in F.items()}).sort_index().astype(float);r=c.pct_change()
ret20=c/c.shift(20)-1; med=ret20.median(axis=1); resid=ret20.sub(med,axis=0)
# trend persistence: 20d residual momentum only when 5d and 20d residual agree; scale by vol
ret5=c/c.shift(5)-1; resid5=ret5.sub(ret5.median(axis=1),axis=0)
vol=r.rolling(20).std(); agree=np.sign(resid)==np.sign(resid5)
f=(resid/vol).where(agree,0.0)
ics={1:[],5:[],10:[]}; ns=[]; rows=[]
for i,date in enumerate(c.index[:-10]):
 x=f.loc[date].replace([np.inf,-np.inf],np.nan); ok=x.notna()&c.loc[date].notna()
 if ok.sum()<8:continue
 ns.append(ok.sum())
 for h in ics:
  y=c.shift(-h).loc[date]/c.loc[date]-1; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:ics[h].append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 for s,v in x.items():rows.append({'date':date,'symbol':s,'signal':v})
print('dates',len(ics[1]),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15)
for h,a in ics.items():
 a=np.array(a);print('H',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'n',len(a))
for st in ['2025-01-01','2027-01-01','2028-01-01','2029-01-01','2029-07-01']:
 a=[]
 for date in c.index[c.index>=st][:-10]:
  x=f.loc[date];y=c.shift(-1).loc[date]/c.loc[date]-1;z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);print(st,len(a),a.mean(),a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
pd.DataFrame(rows).to_csv('scripts/miner_2_20300502_agreement_residual_momentum_signal.csv',index=False)
