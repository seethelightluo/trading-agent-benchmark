import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
d=pd.read_csv('../persistent/index_data/DXY.csv'); d.date=pd.to_datetime(d.date); dx=d.set_index('date').close.astype(float).reindex(P.index).ffill(); dr=np.log(dx).diff()
rows=[]; sig=[]
for i in range(90,len(P)-21):
 rr=R.iloc[i-59:i+1]; zdx=dr.iloc[i-59:i+1]
 b=rr.apply(lambda x:x.cov(zdx)/(zdx.var()+1e-12))
 resid=R.iloc[i-4:i+1].sum()-b*dr.iloc[i-4:i+1].sum(); vol=rr.std()+1e-12
 disp=rr.mean(axis=1).std()+1e-12
 f=(-resid/(vol*np.sqrt(5)))*(1+disp/(rr.mean(axis=1).abs().mean()+1e-12)).clip(1,2)
 f=f.replace([np.inf,-np.inf],np.nan)
 for h in (5,10):
  y=P.iloc[i+h]/P.iloc[i]-1; q=pd.DataFrame({'f':f,'y':y}).dropna()
  if len(q)>=8 and q.f.nunique()>1: rows.append((P.index[i],h,len(q),q.f.corr(q.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna(); x.date=pd.to_datetime(x.date)
print('universe',len(U),'data',P.index.min().date(),P.index.max().date(),'dates',x.date.nunique())
for h in (5,10):
 q=x[x.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('H',h,'obs',len(q),'meanN',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,6),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),6))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-12-31'),('2032-01-01','2032-02-18'),('2031-08-01','2032-02-18')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('REG',a,b,len(w),round(w.ic.mean(),6) if len(w) else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_2_20320219_dxy_neutral_reversal_reval_signal.csv',index=False)
