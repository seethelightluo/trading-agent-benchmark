import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=2600); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); n=len(P); F=np.full((n,15),np.nan)
for i in range(60,n):
 r=R.iloc[:i+1]; res=r.sub(r.median(axis=1),axis=0); rev=res.iloc[i-4:i+1].sum(); v=res.iloc[i-19:i+1].std()*np.sqrt(20); b=res.iloc[i-59:i-19].std()*np.sqrt(40); F[i]=(-rev/(v+1e-12)*np.clip(v/(b+1e-12),.75,1.75)).values
# precompute realized daily cross-sectional IC of raw factor vs 5d forward return
ic=np.full(n,np.nan)
for i in range(60,n-5):
 z=pd.DataFrame({'f':F[i],'y':(P.iloc[i+5]/P.iloc[i]-1).values}).dropna()
 if len(z)>=8 and z.f.nunique()>1: ic[i]=z.f.corr(z.y,method='spearman')
G=F.copy()
for i in range(130,n-5):
 h=ic[i-65:i-4]; h=h[np.isfinite(h)]; gate=1 if len(h)<20 or h.mean()>=0 else -1; G[i]=F[i]*gate
rows=[]; sig=[]
for i in range(130,n-5):
 z=pd.DataFrame({'f':G[i],'y':(P.iloc[i+5]/P.iloc[i]-1).values}).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
  sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in zip(U,G[i])]
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('candidate=adaptive_vol_shock_residual_reversal_5d','assets',15,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-02-19')]:
 q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else np.nan)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'cutoff',P.index[-1].date())
pd.DataFrame(sig).to_csv('scripts/miner_1_20310220_adaptive_vol_shock_residual_reversal_5d_signal.csv',index=False)
