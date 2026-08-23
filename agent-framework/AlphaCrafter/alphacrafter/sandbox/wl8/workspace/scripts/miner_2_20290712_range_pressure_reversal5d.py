import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 try: d=get_stock_daily_data(s,days=4000)
 except Exception as e: print('skip',s,str(e)); continue
 if d is not None and len(d)>250:
  d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
  rng=(d.high-d.low).replace(0,np.nan); clv=((d.close-d.low)/rng-0.5).clip(-1,1)
  r=np.log(d.close).diff(); pressure=clv.rolling(5).mean().shift(1); rev=-r.rolling(5).sum().shift(1)
  d['factor']=(rev*(1+pressure.abs())).replace([np.inf,-np.inf],np.nan); d['fwd1']=np.log(d.close).shift(-1)-np.log(d.close)
  for h in [3,5,10]: d[f'fwd{h}']=np.log(d.close).shift(-h)-np.log(d.close)
  frames[s]=d.reset_index()[['date','factor','fwd1','fwd3','fwd5','fwd10']]
long=pd.concat([x.assign(symbol=s) for s,x in frames.items()],ignore_index=True)
rows=[]
for date,g in long.groupby('date'):
 for h in [1,3,5,10]:
  z=g[['factor',f'fwd{h}']].dropna()
  if len(z)>=8: rows.append({'date':date,'n':len(z),f'ic{h}':z.factor.corr(z[f'fwd{h}'])})
r=pd.DataFrame(rows); print('dates',len(r),'avg_n',r.n.mean(),'coverage',long.factor.notna().groupby(long.date).mean().mean(),'symbols',len(frames))
for h in [1,3,5,10]:
 x=r.set_index('date')[f'ic{h}'].dropna(); print(h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'n',len(x),'recent180',x.tail(180).mean(),x.tail(180).mean()/x.tail(180).std(ddof=1),'recent360',x.tail(360).mean(),x.tail(360).mean()/x.tail(360).std(ddof=1))
long.to_csv('scripts/miner_2_20290712_range_pressure_reversal5d_signal.csv',index=False); print('artifact_rows',len(long))
p=long.dropna(subset=['factor']).pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean().mean())
