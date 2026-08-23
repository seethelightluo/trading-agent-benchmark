import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=5000); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index(); R=np.log(P).diff(); rows=[]
for i in range(120,len(P)-5):
 r20=R.iloc[i-19:i+1].sum(); med=r20.median(); vol=R.iloc[i-59:i+1].std()+1e-12
 down=R.iloc[i-39:i+1].where(R.iloc[i-39:i+1]<0).std(); shock=(down/vol).clip(0.5,2.0)
 f=-(r20-med)/vol*shock; y=P.iloc[i+5]/P.iloc[i]-1
 z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
print('dates',len(rows),'mean_n',np.mean([x[1] for x in rows]),'coverage',np.mean([x[1] for x in rows])/15)
ic=np.array([x[2] for x in rows]); print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0))
print('period',P.index.min().date(),P.index.max().date())
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2031-04-16')]:
 q=[x[2] for x in rows if a<=str(x[0].date())<=b]; print(a,b,len(q),np.mean(q) if q else np.nan)
S=pd.DataFrame({s:R[s] for s in U}); print('signal artifact not written: exploratory validation only')
