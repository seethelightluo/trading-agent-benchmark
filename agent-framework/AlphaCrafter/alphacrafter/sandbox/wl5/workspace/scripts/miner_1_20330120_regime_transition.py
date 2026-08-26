import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4600); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); vc=[c for c in v.columns if c.lower() in ('close','value','vix')][0]; V=v.set_index('date')[vc].astype(float).reindex(P.index).ffill()
rows=[]; sig=[]
for i in range(130,len(P)-11):
 r=R.iloc[:i]; m=r.iloc[-60:].mean(axis=1)
 # market regime transition: VIX level shock versus trailing distribution
 vz=(V.iloc[i-1]-V.iloc[i-21:i-1].mean())/(V.iloc[i-61:i-1].std()+1e-12)
 dv=V.iloc[i-1]-V.iloc[i-6]
 # in rising stress use reversal; in calm use short relative momentum
 stress=float(np.clip(0.5+0.25*vz+0.08*dv,0,1))
 rel=r.iloc[-10:].sum()-r.iloc[-10:].mean(axis=1).iloc[-1] # incorrect scalar broadcast okay
 # recent asset return, blend stress reversal and calm momentum
 recent=r.iloc[-10:].sum()
 f=((2*stress-1)*(-recent)+(1-stress)*recent).rank(pct=True)
 for s,x in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(x)})
 y=P.iloc[i+10]/P.iloc[i]-1; q=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((P.index[i],len(q),q.f.corr(q.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date())
print('horizon',10,'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-01-19')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6),'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_1_20330120_regime_transition_signal.csv',index=False)
