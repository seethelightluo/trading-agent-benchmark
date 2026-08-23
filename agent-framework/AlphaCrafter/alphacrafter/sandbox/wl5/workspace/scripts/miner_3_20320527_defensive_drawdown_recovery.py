import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
for i in range(120,len(P)-11):
 r=R.iloc[:i]
 # Defensive recovery: favor assets with contained recent risk and moderate drawdown recovery,
 # while avoiding a pure trend signal. All inputs end before decision date.
 v20=r.iloc[-20:].std(); v90=r.iloc[-90:].std();
 dd=(P.iloc[:i].iloc[-1]/P.iloc[:i].rolling(120).max().iloc[-1]-1)
 rec=r.iloc[-10:].sum()
 # low-volatility residual plus recovery from drawdown; ranks make cross-asset scale comparable
 f=(-0.65*(v20/(v90+1e-12)).rank(pct=True) + 0.35*rec.rank(pct=True) - 0.20*dd.rank(pct=True))
 # In high common volatility, emphasize the low-vol component; causal regime adjustment
 common=r.iloc[-20:].mean(axis=1).std(); longcommon=r.iloc[-120:].mean(axis=1).std()+1e-12
 stress=np.clip(common/longcommon,0.5,2.0)
 f=(-0.65*stress*(v20/(v90+1e-12)).rank(pct=True) + 0.35*rec.rank(pct=True) - 0.20*dd.rank(pct=True)).rank(pct=True)
 for s,v in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v)})
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date())
print('horizon',10,'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-05-26')]:
 w=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(w),'IC',round(w.ic.mean(),6),'ICIR',round(w.ic.mean()/w.ic.std(ddof=1),6) if len(w)>1 else np.nan)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20320527_defensive_drawdown_recovery_signal.csv',index=False)
