import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=5000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
# Novel factor: 30d risk-adjusted momentum, rewarded only when 90d trend agrees;
# subtract cross-sectional median to emphasize relative trend while preserving rank direction.
for i in range(150,len(P)-21):
 r=R.iloc[:i]; dv=r.iloc[-90:].where(r.iloc[-90:]<0).std()*np.sqrt(252)
 dv=dv.replace(0,np.nan).fillna(r.iloc[-90:].std()*np.sqrt(252)).replace(0,np.nan)+1e-12
 raw=r.iloc[-30:].sum()/dv
 agree=np.sign(r.iloc[-90:].sum())
 f=(raw*agree).replace([np.inf,-np.inf],np.nan)
 f=f-f.median()
 f=f.rank(pct=True)
 for s,x in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(x) if pd.notna(x) else np.nan})
 y=P.iloc[i+20]/P.iloc[i]-1; q=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((P.index[i],len(q),q.f.corr(q.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date())
print('horizon',20,'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-04-27')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6),'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_2_20330428_agree_risk_momentum_20d_signal.csv',index=False)
