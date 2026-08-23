import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
# Cross-asset dispersion gated short reversal: recent 3d residual losses are
# reversed only when trailing cross-sectional dispersion is elevated; otherwise
# signal is attenuated. All inputs lagged at decision date.
for i in range(65,len(P)-11):
 h=R.iloc[i-60:i]; m=h.SPX; beta=h.apply(lambda x:x.cov(m)/(m.var()+1e-12))
 resid=R.iloc[i-3:i].sum()-beta*R.iloc[i-3:i].sum().SPX
 vol=R.iloc[i-20:i].std()+1e-12; base=-resid/vol
 disp=R.iloc[i-20:i].std(axis=1).mean(); ref=R.iloc[i-120:i-20].std(axis=1).mean()
 gate=np.clip(disp/(ref+1e-12),0.6,1.4); f=base*gate
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
 for horizon in (5,10):
  y=P.iloc[i+horizon]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],horizon,len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna(); print('universe',15,'usable_dates',x.date.nunique(),'data_end',P.index.max().date())
for h in (5,10):
 q=x[x.h==h]; print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,5),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-12-31'),('2032-01-01','2032-03-20')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6)); pd.DataFrame(sig).to_csv('scripts/miner_2_20320401_dispersion_reversal_signal.csv',index=False)
