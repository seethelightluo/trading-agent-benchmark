import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; signals=[]
for i in range(61,len(P)-10):
 # causal medium momentum, risk-normalized, activated only when price is above its long trend
 ret20=P.iloc[i]/P.iloc[i-20]-1; vol20=R.iloc[i-19:i+1].std()
 ma60=P.iloc[i-59:i+1].mean(); trend=(P.iloc[i]/ma60-1)
 f=(ret20/(vol20+1e-12))*np.where(trend>0,1.0,0.35)
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 for s,v in f.items(): signals.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v)})
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('assets',len(C),'rows',len(P),'dates',len(x),'meanN',x.n.mean(),'coverage',x.n.sum()/(len(x)*15),'IC',m,'ICIR',m/sd,'hit',(x.ic>0).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-08-07')]:
 z=x[(x.date>=a)&(x.date<=b)]; print(a,b,'dates',len(z),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1) if len(z)>2 else np.nan,'hit',(z.ic>0).mean() if len(z) else np.nan)
S=pd.DataFrame(signals).pivot(index='date',columns='symbol',values='signal'); turn=S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(); print('rank_turnover_proxy',turn)
pd.DataFrame(signals).to_csv('scripts/miner_3_20300808_trend_breakout_persistence_signal.csv',index=False)
