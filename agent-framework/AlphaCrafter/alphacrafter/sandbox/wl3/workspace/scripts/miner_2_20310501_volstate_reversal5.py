import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x):break
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change()
# Volatility-state-conditioned short reversal: recent losses rebound, but favor assets
# whose 20d volatility is below the contemporaneous cross-sectional median.
vol=r.rolling(20).std(); state=(vol<vol.median(axis=1),).astype if False else None
f=(-r.rolling(5).sum()/(vol*np.sqrt(5)+1e-12)) * (1.0+0.5*(vol.lt(vol.median(axis=1),axis=0)))
rows=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.std()>0 and z.y.std()>0: rows.append((p.index[i],z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');
print('factor=vol_state_conditioned_reversal5 horizon=10 end',p.index[-1].date())
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15),'IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/(a.ic.std(ddof=1)+1e-12),(a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=a.loc[lo:hi]
 if len(q):print(lo+'-'+hi,len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12)))
rank=f.rank(axis=1,pct=True);print('turnover',float((rank.diff().abs().mean(axis=1)/2).mean()))
f.index.name='date';f.reset_index().to_csv('scripts/miner_2_20310501_volstate_reversal5_signal.csv',index=False)
