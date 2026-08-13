import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=2800) for s in U}; p=pd.concat({s:d.set_index('date').close for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=p.pct_change();
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); bench=r.mean(axis=1).rolling(20).sum(); out=[]; sig=[]
for t in range(180,len(p)-10):
 h=disp.iloc[t-120:t].dropna()
 if len(h)<100 or not np.isfinite(disp.iloc[t]) or not np.isfinite(bench.iloc[t]): continue
 active=disp.iloc[t]>=np.nanpercentile(h,50) or bench.iloc[t]<0
 if not active: continue
 x=r.iloc[t-40:t]; down=x.where(x<0,0).pow(2).mean().pow(.5)*np.sqrt(40)
 ret=p.iloc[t]/p.iloc[t-20]-1; f=ret/(down+1e-8)
 fr=p.iloc[t+10]/p.iloc[t]-1; z=pd.concat([f.rename('f'),fr.rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: out.append((p.index[t],z.f.corr(z.r),len(z),z.f.rank().corr(z.r.rank())))
 for s,v in f.items(): sig.append((p.index[t],s,v))
o=pd.DataFrame(out,columns=['date','ic','n','rankic']).set_index('date');
for name,x in [('all',o),('recent',o.loc['2029-01-01':'2032-08-31'])]:
 q=x.ic.dropna(); print(name,'dates',len(x),'avg_n',x.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean(),'rankIC',x.rankic.mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=o.loc[a:b].ic.dropna(); print(a+'-'+b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
print('active coverage',len(o)/(len(p)-190),'signal rows',len(sig))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20320930_stress_downside_trend_signal.csv',index=False)
