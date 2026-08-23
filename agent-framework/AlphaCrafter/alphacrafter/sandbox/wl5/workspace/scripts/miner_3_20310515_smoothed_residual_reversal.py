import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); M=R.mean(axis=1)
rows=[]; sig=[]
for i in range(140,len(P)-21):
 rr=R.iloc[i-119:i+1]; m=M.iloc[i-119:i+1]
 beta=rr.apply(lambda x:x.cov(m),axis=0)/(m.var()+1e-12)
 r3=R.iloc[i-2:i+1].sum(); r10=R.iloc[i-9:i+1].sum(); r20=R.iloc[i-19:i+1].sum(); m10=M.iloc[i-9:i+1].sum(); m20=M.iloc[i-19:i+1].sum()
 resid3=r3-beta*m.iloc[i-2:i+1].sum(); resid10=r10-beta*m10; resid20=r20-beta*m20
 v=rr.std()+1e-12
 # smooth fast/medium residual reversal, with mild stress amplification
 raw=-(0.25*resid3/np.sqrt(3)+0.45*resid10/np.sqrt(10)+0.30*resid20/np.sqrt(20))/v
 stress=np.clip((rr.std(axis=1).mean()/(R.iloc[i-239:i+1].std(axis=1).mean()+1e-12)-0.9),0,0.8) if i>=239 else 0
 f=raw*(1+0.25*stress)
 for h in (5,10):
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],h,len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in (5,10):
 q=x[x.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-05-15')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6))
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20310515_smoothed_residual_reversal_5d_signal.csv',index=False)
