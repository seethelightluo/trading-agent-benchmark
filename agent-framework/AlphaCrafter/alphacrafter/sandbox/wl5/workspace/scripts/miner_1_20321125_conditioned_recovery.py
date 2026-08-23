import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4600); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sig=[]
for i in range(100,len(P)-11):
 r=R.iloc[:i]; w=r.iloc[-60:]; recent=r.iloc[-15:]
 # Recovery: recent loss scaled by downside volatility, conditioned on medium trend
 loss=recent.sum(); down=w.clip(upper=0).abs().mean()+1e-12
 trend=r.iloc[-60:-15].sum()
 f=(-loss/down) * (1+0.35*np.tanh(-trend/0.10))
 f=f.rank(pct=True)
 for s,v in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v)})
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('universe',15,'usable_dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,5),'data_end',P.index.max().date())
print('horizon',10,'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-11-24')]:
 q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else np.nan)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.diff().abs().mean(axis=1).dropna().mean(),6)); pd.DataFrame(sig).to_csv('scripts/miner_1_20321125_conditioned_recovery_signal.csv',index=False)
