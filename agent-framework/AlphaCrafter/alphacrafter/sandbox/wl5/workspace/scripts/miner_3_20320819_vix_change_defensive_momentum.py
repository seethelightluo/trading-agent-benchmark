import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.astype(float).reindex(P.index).ffill()
rows=[]; sig=[]
for i in range(130,len(P)-11):
 r=R.iloc[:i]; mom=r.iloc[-20:].sum(); vol=r.iloc[-60:].std()+1e-12
 vz=(v.iloc[i-1]-v.iloc[i-6:i-1].mean())/(v.iloc[i-60:i-1].std()+1e-12)
 # trend following in calm/rising-risk-off, but contrarian after extreme VIX jump
 f=mom/vol
 if vz>1: f=-f*(1+0.25*np.clip(vz,0,3))
 elif vz<-1: f=f*1.15
 f=f.rank(pct=True)
 for s,x in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(x)})
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date())
print('horizon',10,'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-08-18')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6) if len(w) else None,'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20320819_vix_change_defensive_momentum_signal.csv',index=False)
