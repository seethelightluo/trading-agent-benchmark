import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x):break
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff()
mom=r.rolling(30,min_periods=20).sum();vol=r.rolling(30,min_periods=20).std();eff=mom.abs()/r.abs().rolling(30,min_periods=20).sum();f=(mom/vol*eff).shift(1)
rows=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i].rename('f'),np.log(p.iloc[i+10]/p.iloc[i]).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1:rows.append((p.index[i],z.f.corr(z.y,method='spearman'),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna();print('shape',p.shape,'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/len(p.columns));print('H10 IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=a.loc[lo:hi]
 if len(q):print(lo+'-'+hi,len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
q=a.tail(120);print('recent120',q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1));print('turnover',(f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean());f.index.name='date';f.to_csv('scripts/miner_3_20310807_quiet_trend_quality_signal.csv');a.to_csv('scripts/miner_3_20310807_quiet_trend_quality_ic.csv')
