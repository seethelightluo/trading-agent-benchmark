import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cs={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);cs[s]=d.set_index('date').close
P=pd.DataFrame(cs).sort_index(); r=P.pct_change()
q=-(r.rolling(3,min_periods=3).sum())/(r.rolling(20,min_periods=15).std()*np.sqrt(3)+1e-12)
disp=r.rolling(3,min_periods=3).std().mean(axis=1)
med=disp.rolling(60,min_periods=30).median()
active=(disp>1.20*med).shift(1)
sig=q.shift(1).where(active, np.nan); sig=sig.rank(axis=1,pct=True).sub(.5)
ics={1:[],5:[],10:[],20:[]}; ns=[]; dates=[]
for dt in sig.index:
 v=sig.loc[dt].notna()
 if v.sum()<8: continue
 dates.append(dt);ns.append(v.sum())
 for h in ics:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; z=v&y.notna()
  ics[h].append(sig.loc[dt,z].corr(y[z],method='spearman') if z.sum()>=8 else np.nan)
for h,a in ics.items():
 a=pd.Series(a).dropna(); print('h',h,'dates',len(a),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('rows',len(P),'assets',len(P.columns),'active_dates',len(dates),'avg_n %.2f'%np.mean(ns),'coverage %.5f'%sig.notna().mean().mean(),'turnover %.5f'%sig.diff().abs().mean().mean())
a=pd.Series(ics[1],index=pd.to_datetime(dates)).dropna()
for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]: print('regime',round(a.iloc[i:j].mean(),8))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310616_disp12_signal.csv',index=False)
pd.DataFrame({'date':a.index,'ic':a.values}).to_csv('scripts/miner_2_20310616_disp12_ic.csv',index=False)
