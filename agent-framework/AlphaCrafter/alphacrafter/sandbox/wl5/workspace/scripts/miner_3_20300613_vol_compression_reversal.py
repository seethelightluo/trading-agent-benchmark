import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
rows=[]
for i in range(25,len(P)-10):
 f=(-P.iloc[i]/P.iloc[i-5]+1)/(R.iloc[i-4:i+1].std()/(R.iloc[i-19:i+1].std()+1e-12)+.25)
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('assets',len(C),'rows',len(P),'dates',len(x),'meanN',x.n.mean(),'coverage',x.n.sum()/(len(x)*15),'IC',m,'ICIR_daily',m/sd,'hit',(x.ic>0).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-06-13')]:
 z=x[(x.date>=a)&(x.date<=b)]; print(a,b,'dates',len(z),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1) if len(z)>2 else np.nan,'hit',(z.ic>0).mean() if len(z) else np.nan)
out=[]
for i in range(25,len(P)-10):
 f=(-P.iloc[i]/P.iloc[i-5]+1)/(R.iloc[i-4:i+1].std()/(R.iloc[i-19:i+1].std()+1e-12)+.25)
 for s,v in f.items(): out.append({'date':str(P.index[i].date()),'symbol':s,'signal':v})
pd.DataFrame(out).to_csv('scripts/miner_3_20300613_vol_compression_reversal_signal.csv',index=False)
