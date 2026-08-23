import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sigs=[]
for i in range(61,len(P)-10):
 r20=P.iloc[i]/P.iloc[i-20]-1
 # reward persistent gains but penalize downside risk; cross-sectional regime neutralization
 dn=R.iloc[i-39:i+1].clip(upper=0).std()
 vol=R.iloc[i-39:i+1].std()
 f=(r20/(dn+1e-12))*np.sqrt(np.clip(vol.median()/(vol+1e-12),0.25,4.0))
 # avoid stale/invalid series and neutralize location for cross-asset comparability
 y=P.iloc[i+10]/P.iloc[i]-1
 z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 for s,v in f.items(): sigs.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v)})
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('assets',15,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-08-21')]:
 z=x[(x.date>=a)&(x.date<=b)]; print(a,b,'dates',len(z),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),5))
S=pd.DataFrame(sigs).pivot(index='date',columns='symbol',values='signal')
print('rank_turnover_proxy',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
# decay on same factor observations
for h in [5,10,20]:
 rr=[]
 for i in range(61,len(P)-h):
  r20=P.iloc[i]/P.iloc[i-20]-1; dn=R.iloc[i-39:i+1].clip(upper=0).std(); vol=R.iloc[i-39:i+1].std(); f=(r20/(dn+1e-12))*np.sqrt(np.clip(vol.median()/(vol+1e-12),.25,4))
  y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y,method='spearman'))
 print('decay',h,'dates',len(rr),'IC',round(np.nanmean(rr),6),'ICIR',round(np.nanmean(rr)/np.nanstd(rr,ddof=1),6))
pd.DataFrame(sigs).to_csv('scripts/miner_3_20300905_downside_adjusted_momentum_signal.csv',index=False)
