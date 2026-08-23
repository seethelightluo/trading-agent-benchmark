import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); M=R.mean(axis=1)
vix=get_index_daily_data('VIX',days=4000); vix.date=pd.to_datetime(vix.date); V=vix.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float).reindex(P.index).ffill()
rows=[]; sig=[]
for i in range(280,len(P)-21):
 rr=R.iloc[i-59:i+1]; m=M.iloc[i-59:i+1]
 beta=rr.apply(lambda x:x.cov(m),axis=0)/(m.var()+1e-12)
 resid=R.iloc[i-4:i+1].sum()-beta*m.iloc[i-4:i+1].sum(); vol=rr.std()+1e-12
 # causal stress percentile and contemporaneous cross-asset dispersion, both observed at i
 vp=V.iloc[max(0,i-252):i].rank(pct=True).iloc[-1] if V.iloc[i-252:i].notna().sum()>100 else .5
 disp=R.iloc[i].std(); dh=R.iloc[i-59:i+1].std(axis=1).median()
 gate=1+0.55*max(vp-.60,0)/.40+0.35*max(disp/(dh+1e-12)-1,0)
 f=-resid/(vol*np.sqrt(5))*gate
 for h in (5,10):
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],h,len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in (5,10):
 q=x[x.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-07-09')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6))
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_1_20310710_stress_dispersion_vix_residual_5d_signal.csv',index=False)
