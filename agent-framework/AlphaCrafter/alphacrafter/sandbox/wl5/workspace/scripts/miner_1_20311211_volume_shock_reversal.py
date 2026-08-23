import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); d=d.sort_values('date').drop_duplicates('date').set_index('date')
 P[s]=d.close.astype(float); V[s]=d.volume.astype(float)
P=pd.DataFrame(P).sort_index(); V=pd.DataFrame(V).reindex(P.index); R=np.log(P).diff()
rows=[]; sig=[]
# Volume-shock reversal: fade relative 5d move, amplified by elevated volume, normalized by 20d volatility.
for i in range(25,len(P)-21):
 r=R.iloc[i-4:i].sum(); med=r.median(); vol=R.iloc[i-20:i].std().replace(0,np.nan)+1e-12
 vs=(V.iloc[i-4:i].mean()/(V.iloc[i-24:i-4].mean()+1e-12)).clip(0.5,3.0)
 f=(-(r-med)/vol)*np.sqrt(vs)
 f=f.replace([np.inf,-np.inf],np.nan)
 for h in (5,10):
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],h,len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna(); print('universe',15,'usable_dates',x.date.nunique())
for h in (5,10):
 q=x[x.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1); print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-12-10')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6)); pd.DataFrame(sig).to_csv('scripts/miner_1_20311211_volume_shock_reversal_signal.csv',index=False)
