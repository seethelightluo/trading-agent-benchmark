import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
# robust common component: cross-sectional median, then short residual shock / residual downside risk
M=R.median(axis=1); rows=[]; sig=[]
for i in range(45,len(P)-10):
 res=R.sub(M,axis=0)
 shock=res.iloc[i-4:i+1].sum()
 downside=res.iloc[i-29:i+1].where(res.iloc[i-29:i+1]<0).std()*np.sqrt(30)
 f=-shock/(downside+1e-12)
 y=P.iloc[i+10]/P.iloc[i]-1
 z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 for s,a in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(a)})
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('candidate=robust_residual_reversal_5d','assets',15,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-11-14')]:
 q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else np.nan)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [5,20]:
 qv=[]; res=R.sub(M,axis=0)
 for i in range(45,len(P)-h):
  shock=res.iloc[i-4:i+1].sum(); downside=res.iloc[i-29:i+1].where(res.iloc[i-29:i+1]<0).std()*np.sqrt(30); f=-shock/(downside+1e-12); y=P.iloc[i+h]/P.iloc[i]-1
  z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1:qv.append(z.f.corr(z.y,method='spearman'))
 print('decay',h,'dates',len(qv),'IC',round(np.nanmean(qv),6),'ICIR',round(np.nanmean(qv)/np.nanstd(qv,ddof=1),6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20301114_robust_residual_reversal_signal.csv',index=False)
