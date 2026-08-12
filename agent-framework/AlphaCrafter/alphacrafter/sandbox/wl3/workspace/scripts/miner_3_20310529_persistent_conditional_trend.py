import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x):break
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();r10=p.pct_change(10);r20=p.pct_change(20);v=r.rolling(40).std();breadth=(r20>0).mean(axis=1)
# Novel smoother activation: supportive breadth and median trend must persist for 3 completed sessions.
support=((r20.median(axis=1)>0)&(breadth>=0.5)).astype(int)
gate=(support.rolling(3,min_periods=3).sum()==3).astype(float)
f=(r10/(v*np.sqrt(10)+1e-12)).where(gate.shift(1).eq(1))
rows=[]
for i in range(len(p)-11):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('shape',p.shape,'active',gate.mean(),'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/(a.ic.std(ddof=1)+1e-12),(a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=a.loc[lo:hi]
 if len(q): print(lo+'-'+hi,len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12)))
for h in [3,5,10,20]:
 rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y))
 print('H',h,'dates',len(rr),'IC',np.nanmean(rr),'ICIR',np.nanmean(rr)/(np.nanstd(rr,ddof=1)+1e-12))
print('turnover',(f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),'last_date',p.index[-1].date())
f.index.name='date';f.reset_index().to_csv('scripts/miner_3_20310529_persistent_conditional_trend_signal.csv',index=False)
