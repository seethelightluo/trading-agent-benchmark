import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
# Smooth 5-day contrarian shock, normalized by 30d volatility; tail asymmetry is a mild stabilizer.
for i in range(90,len(P)-21):
 r5=R.iloc[i-4:i+1].sum(); v30=R.iloc[i-29:i+1].std()+1e-12
 tail=(R.iloc[i-29:i+1].clip(upper=0).abs().mean()+1e-12)/v30
 f=(-r5/v30)*(0.90+0.10*tail)
 f=f.replace([np.inf,-np.inf],np.nan)
 for h in (5,10):
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],h,len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in (5,10):
 q=x[x.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-10-15')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6))
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20311016_smooth_tail_reversal_signal.csv',index=False)
