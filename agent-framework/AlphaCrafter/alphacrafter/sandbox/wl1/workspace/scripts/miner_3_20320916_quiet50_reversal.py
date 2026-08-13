import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=2800) for s in U}; p=pd.concat({s:d.set_index('date').close for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=p.pct_change();
# dispersion: cross-sectional daily dispersion smoothed over 20 sessions
cs=r.std(axis=1); disp=cs.rolling(20,min_periods=15).mean(); out=[]
for t in range(180,len(p)-10):
 hist=disp.iloc[t-120:t].dropna()
 if len(hist)<100 or not np.isfinite(disp.iloc[t]): continue
 if disp.iloc[t] > np.nanpercentile(hist,50): continue
 ret40=p.iloc[t]/p.iloc[t-40]-1; resid=ret40-ret40.mean(); vol=r.iloc[t-40:t].std()*np.sqrt(40); f=-resid/vol
 fr=p.iloc[t+10]/p.iloc[t]-1; z=pd.concat([f.rename('f'),fr.rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: out.append((p.index[t],z.f.corr(z.r),len(z),z.f.rank().corr(z.r.rank())))
o=pd.DataFrame(out,columns=['date','ic','n','rankic']).set_index('date');
for name,x in [('all',o),('recent',o.loc['2029-01-01':'2032-08-31'])]:
 ic=x.ic.dropna(); print(name,'dates',len(x),'avg_n',x.n.mean() if len(x) else 0,'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'rankIC',x.rankic.mean())
print('periods')
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 x=o.loc[a:b].ic.dropna(); print(a,b,len(x),x.mean(),x.mean()/x.std() if len(x)>1 else np.nan)
print('coverage',len(o)/(len(p)-190))
# signal artifact for reproducibility
sig=[]
for t in range(180,len(p)-10):
 hist=disp.iloc[t-120:t].dropna()
 if len(hist)>=100 and disp.iloc[t]<=np.nanpercentile(hist,50):
  rr=p.iloc[t]/p.iloc[t-40]-1; vv=r.iloc[t-40:t].std()*np.sqrt(40); f=-(rr-rr.mean())/vv
  for s,v in f.items(): sig.append((p.index[t],s,v))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20320916_quiet50_reversal_signal.csv',index=False)
