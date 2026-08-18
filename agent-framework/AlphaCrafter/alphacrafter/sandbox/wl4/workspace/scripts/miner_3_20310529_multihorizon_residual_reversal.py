import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try: D[s]=get_index_daily_data(s,days=4000)
 except Exception: D[s]=get_stock_daily_data(s,days=4000)
px=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=px.pct_change()
# Continuous multi-horizon residual reversal: blend 5d and 20d relative returns,
# scaled by 20d realized volatility, with a mild 60d trend anchor.
cs5=r.rolling(5).sum().sub(r.rolling(5).sum().median(axis=1),axis=0)
cs20=r.rolling(20).sum().sub(r.rolling(20).sum().median(axis=1),axis=0)
vol=r.rolling(20).std().replace(0,np.nan)
f=(-(0.70*cs5+0.30*cs20)/(vol+1e-12)).shift(1)
rows=[]
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; a=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],fw.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:a.append((px.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 res=pd.DataFrame(a,columns=['date','ic','n']).set_index('date'); ic=res.ic
 if h==10:
  turn=[]
  for i in range(1,len(f)):
   turn.append((f.iloc[i].rank(pct=True)-f.iloc[i-1].rank(pct=True)).abs().mean())
  print('factor=multihorizon_residual_reversal dates=%d avg_n=%.2f coverage=%.4f'%(len(res),res.n.mean(),res.n.mean()/15))
  print('H10 IC=%.6f ICIR=%.6f hit=%.4f turnover=%.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),np.nanmean(turn)))
  for w in [365,730,1095]:
   q=ic.tail(w); print('recent%d IC=%.6f ICIR=%.6f hit=%.4f dates=%d'%(w,q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q)))
 else: print('H%d IC=%.6f ICIR=%.6f dates=%d'%(h,ic.mean(),ic.mean()/ic.std(ddof=1),len(ic)))
