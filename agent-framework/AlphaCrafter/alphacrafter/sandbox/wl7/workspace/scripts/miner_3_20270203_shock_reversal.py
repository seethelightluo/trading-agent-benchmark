import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=get_stock_daily_data(s,1900)
 if d is not None and len(d): p[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(p).sort_index(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
# short-term shock reversal: selloff relative to volatility, damped when longer trend is strongly negative
shock=-(p/p.shift(5)-1)/(v*np.sqrt(5)+1e-6)
trend=(p/p.shift(60)-1)
f=shock*(1-0.25*np.tanh(-trend*4)).shift(1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt].rename('x'),r.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.x.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=a.ic
print({'dates':len(q),'avg_n':a.n.mean(),'ic':q.mean(),'icir':q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit':(q>0).mean(),'coverage':f.notna().sum().sum()/(f.shape[0]*len(U)),'turnover':f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()})
for h in [5,10,20]:
 y=p.pct_change(h).shift(-h); zics=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:zics.append(z.x.corr(z.y))
 print('h',h,'ic',np.nanmean(zics),'n',len(zics))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 z=q.loc[lo:hi]; print(lo,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270203_shock_reversal_signal.csv',index=False)
