import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); M=R.mean(axis=1)
rows=[]; sig=[]
for i in range(120,len(P)-21):
 rr=R.iloc[i-59:i+1]; m=M.iloc[i-59:i+1]
 beta=rr.apply(lambda x:x.cov(m),axis=0)/(m.var()+1e-12)
 shock=R.iloc[i-4:i+1].sum()-beta*m.iloc[i-4:i+1].sum()
 vol=rr.std()+1e-12
 disp=R.iloc[i-19:i+1].std(axis=1).mean()
 # Adaptive reversal: residual shock is more useful in ordinary/high dispersion,
 # while avoiding excessive amplification in the most dislocated sessions.
 q=R.iloc[i-59:i+1].std(axis=1).rolling(20).mean().iloc[-1]
 gate=np.clip(disp/(q+1e-12),0.75,1.45)
 f=-shock/(vol*np.sqrt(5))*gate
 for h in (5,10):
  y=P.iloc[i+h]/P.iloc[i]-1
  z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],h,len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in (5,10):
 q=x[x.h==h]; mu=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,5),'IC',round(mu,6),'ICIR',round(mu/sd,6),'hit',round((q.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-09-16')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6))
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal')
print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_2_20310918_dispersion_adaptive_residual_reversal_5d_signal.csv',index=False)
