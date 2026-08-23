import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}; vols={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 d=d.assign(date=pd.to_datetime(d.date)).sort_values('date').drop_duplicates('date').set_index('date')
 prices[s]=d.close.astype(float); vols[s]=d.volume.astype(float) if 'volume' in d else pd.Series(index=d.index,dtype=float)
P=pd.DataFrame(prices).sort_index(); V=pd.DataFrame(vols).reindex(P.index)
R=np.log(P).diff(); out=[]; sig=[]
for i in range(61,len(P)-10):
 r5=P.iloc[i]/P.iloc[i-5]-1
 # volume shock is relative to trailing 20d median, winsorized; only volume-confirmed downside reversal
 vr=(V.iloc[i]/(V.iloc[i-20:i].median()+1e-12)).clip(0.25,4.0)
 v20=R.iloc[i-19:i+1].std(); f=-np.minimum(r5,0)*np.sqrt(vr)/(v20+1e-12)
 y=P.iloc[i+10]/P.iloc[i]-1
 z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: out.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 for s,v in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(v) if np.isfinite(v) else np.nan})
x=pd.DataFrame(out,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('assets',len(U),'rows',len(P),'dates',len(x),'meanN',x.n.mean(),'coverage',x.n.sum()/(len(x)*15),'IC',m,'ICIR',m/sd,'hit',(x.ic>0).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-07-24')]:
 z=x[(x.date>=a)&(x.date<=b)]; print(a,b,'dates',len(z),'IC',z.ic.mean() if len(z) else np.nan,'ICIR',z.ic.mean()/z.ic.std(ddof=1) if len(z)>2 else np.nan,'hit',(z.ic>0).mean() if len(z) else np.nan)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover_proxy',(S.rank(axis=1,pct=True).diff().abs().mean(axis=1)).mean())
pd.DataFrame(sig).to_csv('scripts/miner_2_20300725_volume_confirmed_reversal_signal.csv',index=False)
