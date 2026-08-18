import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2028-01-13'); px={}
for s in U:
 d=get_stock_daily_data(s,days=3000); x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x[x.date<=cut].drop_duplicates('date').set_index('date').close
P=pd.DataFrame(px).sort_index().ffill(); ret=P.pct_change(); neg2=ret.clip(upper=0).pow(2)
down=np.sqrt(neg2.rolling(20,min_periods=20).mean())*np.sqrt(252)
f=-(P/P.shift(5)-1)/(down+1e-8); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); rank=f.rank(axis=1,pct=True)
print('cutoff',P.index.max().date(),'dates',len(r),'instruments',len(U),'mean_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean(),'turn',rank.diff().abs().mean(axis=1).mean())
for h in [5,10,20]:
 rr=[]; ff=P.shift(-h)/P-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr),len(rr))
for name,sub in [('2025-26',r.loc['2025':'2026']),('2027',r.loc['2027-01-01':'2027-12-31'])]: print(name,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1) if len(sub)>1 else np.nan)
print('recent',r.tail(60).ic.mean(),len(r.tail(60)))
