import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}; C={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x.set_index('date').sort_index(); rng=(x.high-x.low).replace(0,np.nan); F[s]=(-(x.close-x.open)/rng); C[s]=x.close
f=pd.concat(F,axis=1).sort_index(); px=pd.concat(C,axis=1).reindex(f.index); sig=f.shift(1); out=[]
for i,dt in enumerate(f.index[:-1]):
 z=pd.concat([sig.loc[dt],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
o=pd.DataFrame(out,columns=['date','ic','n']); print('dates',len(o),'avgN',o.n.mean(),'IC %.6f ICIR %.6f hit %.3f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for lab,m in [('2020-22',o.date.dt.year<=2022),('2023-25',o.date.dt.year.between(2023,2025)),('2026',o.date.dt.year==2026),('2027',o.date.dt.year==2027),('2028',o.date.dt.year==2028),('recent180',o.date>=o.date.max()-pd.Timedelta(days=280))]:
 q=o[m].ic
 if len(q): print(lab,len(q),'%.6f %.6f'%(q.mean(),q.mean()/q.std()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20280629_clv_signal.csv',index=False)
