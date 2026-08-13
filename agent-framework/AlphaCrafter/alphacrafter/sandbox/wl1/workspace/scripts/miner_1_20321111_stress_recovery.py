import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff()
low120=p.rolling(120,min_periods=80).min(); rec=np.log(p/low120)
vol=r.rolling(30,min_periods=15).std()*np.sqrt(252); down=r.where(r<0).rolling(40,min_periods=20).std()*np.sqrt(252)
bench=r.mean(axis=1); disp=r.std(axis=1).rolling(20,min_periods=12).mean(); qdisp=disp.rolling(120,min_periods=60).quantile(.65)
gate=((bench.rolling(20,min_periods=15).sum()<0)|(disp>qdisp)).astype(float)
score=rec.div(down+0.5*vol+1e-6).mul(gate,axis=0).shift(1)
sig=score.rank(axis=1,pct=True)
rows=[]
for dt in sig.index:
 i=p.index.get_loc(dt)
 if i+10>=len(p): continue
 y=p.iloc[i+10]/p.iloc[i]-1
 z=pd.concat([sig.loc[dt].rename('x'),y.rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((dt,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']); o.date=pd.to_datetime(o.date)
print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.n.mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean(),'active',gate.mean())
print('h10 obs',len(o),'IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean()))
for a,b in [(2020,2023),(2024,2026),(2027,2029),(2030,2032)]:
 z=o[(o.date.dt.year>=a)&(o.date.dt.year<=b)].ic
 if len(z)>2: print('period',a,b,'IC %.6f ICIR %.6f n=%d'%(z.mean(),z.mean()/z.std(ddof=1),len(z)))
sig.to_csv('scripts/miner_1_20321111_stress_recovery_signal.csv')
