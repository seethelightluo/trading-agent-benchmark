import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); M=R.mean(axis=1)
# Observation-only VIX, used strictly lagged through signal date
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date)
vix=vix.set_index('date').iloc[:,0].astype(float).reindex(P.index).ffill()
rows=[]; sig=[]
for i in range(80,len(P)-21):
 rr=R.iloc[i-59:i+1]; m=M.iloc[i-59:i+1]
 beta=rr.apply(lambda x:x.cov(m),axis=0)/(m.var()+1e-12)
 # one-day idiosyncratic shock, normalized by recent risk and amplified in stressed regimes
 shock=R.iloc[i]-beta*M.iloc[i]
 vol=rr.std()+1e-12
 downside=rr.where(rr<0).std().fillna(vol)+1e-12
 vp=vix.iloc[:i+1].tail(252); percentile=(vp.rank(pct=True).iloc[-1] if len(vp)>20 else .5)
 stress=1+1.0*max(0,percentile-.60)/.40
 asym=np.clip(downside/vol,.5,2.0)
 f=-shock/vol*asym*stress
 for h in (5,10,20):
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],h,len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
print('data_end',P.index.max().date(),'vix_end',vix.dropna().index.max().date())
for h in (5,10,20):
 q=x[x.h==h]; mean=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'meanN',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,5),'IC',round(mean,6),'ICIR',round(mean/sd,6),'hit',round((q.ic>0).mean(),5))
 for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-05-28')]:
  w=q[(q.date>=a)&(q.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_1_20310529_shock_recovery_signal.csv',index=False)
