import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); d=d.sort_values('date').drop_duplicates('date').set_index('date'); C[s]=d.close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
# Orthogonalized short-term reversal: remove the cross-sectional component
# explained by intermediate (20d) trend, then reverse the 5d idiosyncratic move.
for i in range(80,len(P)-21):
 r5=R.iloc[i-4:i+1].sum(); r20=R.iloc[i-19:i+1].sum(); vol=R.iloc[i-59:i+1].std()+1e-12
 z=pd.DataFrame({'r5':r5,'r20':r20,'vol':vol}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)<8: continue
 x=z.r20-z.r20.mean(); y=z.r5-z.r5.mean(); beta=(x*y).sum()/((x*x).sum()+1e-12)
 resid=y-beta*x
 f=(-resid/(z.vol+1e-12)).clip(-8,8)
 for h in (5,10):
  yy=P.iloc[i+h]/P.iloc[i]-1; q=pd.DataFrame({'f':f,'y':yy}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.f.nunique()>1: rows.append((P.index[i],h,len(q),q.f.corr(q.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(f.get(s,np.nan))} for s in U]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in (5,10):
 q=x[x.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-08-31')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6))
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_1_20310904_orthogonal_reversal_5d_signal.csv',index=False)
