import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={}
for s in U:
 d=get_stock_daily_data(s,days=2600); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); n=len(P); F=np.full((n,15),np.nan)
for i in range(65,n):
 r=R.iloc[:i+1]; res=r.sub(r.median(axis=1),axis=0); tr=res.iloc[i-19:i+1].sum(); v=res.iloc[i-59:i+1].std()*np.sqrt(60); breadth=(res.iloc[i-19:i+1]>0).mean(); disp=res.iloc[i-19:i+1].std(axis=1).mean(); base=res.iloc[i-119:i-1].std(axis=1).mean()
 # continuation only when asset's medium trend agrees with broad cross-section; volatility and dispersion normalization
 F[i]=(tr/(v+1e-12)*(0.5+0.5*breadth)*np.clip(disp/(base+1e-12),.75,1.5)).values
rows=[]; sig=[]
for i in range(120,n-5):
 z=pd.DataFrame({'f':F[i],'y':(P.iloc[i+5]/P.iloc[i]-1).values}).dropna()
 if len(z)>=8 and z.f.nunique()>1:
  rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman'))); sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in zip(U,F[i])]
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('candidate=dispersion_breadth_confirmed_relative_momentum_20d_5d','assets',15,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-02-19')]:
 q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else np.nan)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'cutoff',P.index[-1].date())
for h in [10,20]:
 q=[]
 for i in range(120,n-h):
  z=pd.DataFrame({'f':F[i],'y':(P.iloc[i+h]/P.iloc[i]-1).values}).dropna()
  if len(z)>=8 and z.f.nunique()>1:q.append(z.f.corr(z.y,method='spearman'))
 print('decay',h,'dates',len(q),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6))
pd.DataFrame(sig).to_csv('scripts/miner_1_20310220_dispersion_breadth_confirmed_momentum_signal.csv',index=False)
