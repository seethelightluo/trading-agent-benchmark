import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; closes={}; vols={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x): break
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x.sort_values('date').drop_duplicates('date').set_index('date')
  closes[s]=pd.to_numeric(x.close,errors='coerce'); vols[s]=pd.to_numeric(x.volume,errors='coerce')
p=pd.DataFrame(closes).sort_index().ffill(); v=pd.DataFrame(vols).reindex(p.index).replace(0,np.nan).ffill(); r=p.pct_change()
# Volume-confirmed mean reversion: recent losses/gains are reversed, with
# stronger signal when the move occurred on unusually high participation;
# scale by trailing volatility to compare assets.
ret5=r.rolling(5,min_periods=5).sum(); vol20=r.rolling(20,min_periods=15).std()*np.sqrt(20); vshock=(v/(v.rolling(20,min_periods=15).median()+1e-12)).clip(0.25,4.0)
f=(-ret5/(vol20+1e-12))*np.sqrt(vshock); f=f.shift(1)

def calc(h):
 rows=[]
 for i in range(len(p)-h-1):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h+1]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 return a
for h in [1,5,10,20]:
 a=calc(h); print('H%d dates %d avg_n %.2f coverage %.4f IC %.8f ICIR %.8f hit %.4f'%(h,len(a),a.n.mean(),a.n.sum()/(len(a)*15),a.ic.mean(),a.ic.mean()/(a.ic.std(ddof=1)+1e-12),(a.ic>0).mean()))
a=calc(10)
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=a.loc[lo:hi]
 if len(q): print('REGIME',lo+'-'+hi,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/(q.ic.std(ddof=1)+1e-12))
turn=((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean()); print('shape',p.shape,'last',p.index[-1].date(),'turnover',turn)
f.index.name='date'; f.reset_index().to_csv('scripts/miner_2_20310821_volume_confirmed_reversal_signal.csv',index=False)
