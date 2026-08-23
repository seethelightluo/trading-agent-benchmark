import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index().ffill(); R=np.log(P).diff(); M=R.mean(axis=1)
rows=[]; sig=[]
for i in range(130,len(P)-21):
 rr=R.iloc[i-119:i+1]; m=M.reindex(rr.index)
 beta=rr.apply(lambda x:x.cov(m),axis=0)/(m.var()+1e-12)
 short=R.iloc[i-19:i+1]; ms=m.reindex(short.index).values
 resid=short-pd.DataFrame(ms[:,None]*beta.values[None,:],index=short.index,columns=R.columns)
 # intermediate residual momentum, conditioned by cross-sectional breadth
 mom=resid.iloc[-10:].sum(); vol=rr.std()+1e-12
 breadth=(R.iloc[i-19:i+1].gt(0).mean(axis=1).mean()-0.5)*2
 disp=R.iloc[i-4:i+1].std(axis=1).mean(); base=R.iloc[i-59:i+1].std(axis=1).mean()+1e-12
 gate=np.clip(disp/base-1,-.7,.7)
 # trend continuation when breadth confirms, otherwise reversal of latest shock
 f=mom/(vol*np.sqrt(10))*(1+0.4*gate)*(1+0.35*breadth)-0.30*(-resid.iloc[-1]/vol)*(1-0.25*breadth)
 for h in (5,10):
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],h,len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in (5,10):
 q=x[x.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,6),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),6))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-09-30')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6))
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_2_20311002_breadth_dispersion_residual_momentum_signal.csv',index=False)
