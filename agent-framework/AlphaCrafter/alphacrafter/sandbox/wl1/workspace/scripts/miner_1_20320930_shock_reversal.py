import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=2800) for s in U}; p=pd.concat({s:d.set_index('date').close for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=p.pct_change(); disp=r.std(axis=1).rolling(20,min_periods=15).mean(); out=[]; sig=[]
# High-dispersion short-horizon shock reversal: fade 5d risk-normalized shocks, lagged one completed day
for t in range(180,len(p)-11):
 h=disp.iloc[t-120:t].dropna()
 if len(h)<100 or disp.iloc[t]<=np.nanpercentile(h,50): continue
 rr=p.iloc[t]/p.iloc[t-5]-1; v=r.iloc[t-40:t].std()*np.sqrt(5); f=-rr/v
 fr=p.iloc[t+10]/p.iloc[t]-1; z=pd.concat([f.rename('f'),fr.rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: out.append((p.index[t],z.f.corr(z.r),len(z),z.f.rank().corr(z.r.rank())))
 for s,x in f.items(): sig.append((p.index[t],s,x))
o=pd.DataFrame(out,columns=['date','ic','n','rankic']).set_index('date')
for name,x in [('all',o),('recent',o.loc['2029-01-01':'2032-08-31'])]:
 ic=x.ic.dropna(); print(name,'dates',len(x),'avg_n',x.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'rankIC',x.rankic.mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 x=o.loc[a:b].ic.dropna(); print(a,b,len(x),x.mean(),x.mean()/x.std() if len(x)>1 else np.nan)
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20320930_shock_reversal_signal.csv',index=False)
