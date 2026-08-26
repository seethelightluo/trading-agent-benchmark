import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4700); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); vc=[c for c in v.columns if c.lower() in ('close','value','vix')][0]; V=v.set_index('date')[vc].astype(float).reindex(P.index).ffill()
rows=[]; sig=[]
for i in range(120,len(P)-41):
 r=R.iloc[:i]; ds=r.iloc[-90:].clip(upper=0).std()+1e-12
 base=-r.iloc[-80:].sum()/ds
 # volatility-transition gate: emphasize reversal after a recent cross-asset vol shock
 cv=r.iloc[-10:].std(axis=1).mean(); lv=r.iloc[-60:].std(axis=1).mean()+1e-12
 shock=np.clip(cv/lv-1,0,1.5)/1.5
 breadth=(r.iloc[-20:].sum()>0).mean(); weak=np.clip((.5-breadth)*2,0,1)
 vz=(V.iloc[i-1]-V.iloc[i-61:i].mean())/(V.iloc[i-61:i].std()+1e-12); vix=np.clip(vz,0,2)/2
 f=(base*(1+.55*shock+.35*weak+.35*vix)).replace([np.inf,-np.inf],np.nan).rank(pct=True)
 for s,x in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(x)})
 y=P.iloc[i+40]/P.iloc[i]-1; q=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((P.index[i],len(q),q.f.corr(q.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date())
print('horizon',40,'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2033-04-13')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6),'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20330414_voltransition_reversal_40d_signal.csv',index=False)
