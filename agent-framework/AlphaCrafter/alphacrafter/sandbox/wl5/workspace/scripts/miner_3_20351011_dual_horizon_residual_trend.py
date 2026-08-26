import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
def factor(i):
 # causal dual-horizon trend: residualize each horizon against equal-weight market,
 # then reward persistent 60d trend while penalizing recent 10d reversal; volatility normalize
 r10=R.iloc[i-10:i].sum(); r60=R.iloc[i-60:i].sum(); m10=r10.mean(); m60=r60.mean()
 vol=R.iloc[i-40:i].std().replace(0,np.nan)+1e-12
 f=((r60-m60)/vol)*0.65 + ((r10-m10)/vol)*0.35
 return f.replace([np.inf,-np.inf],np.nan).clip(-10,10)
for i in range(80,len(P)-21):
 f=factor(i)
 for s,v in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v) if pd.notna(v) else np.nan})
 for h in (5,10,20):
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],h,len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna(); print('universe',15,'data_start',P.index.min().date(),'data_end',P.index.max().date(),'usable_dates',x.date.nunique())
for h in (5,10,20):
 q=x[x.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1); print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,6),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),6))
 for a,b in [('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2031-12-31'),('2032-01-01','2035-09-30')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20351011_dual_horizon_residual_trend_signal.csv',index=False)
