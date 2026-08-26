import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is not None and len(d)>100:
  q=d.copy(); q['date']=pd.to_datetime(q['date']); px[s]=q.set_index('date')['close'].astype(float).sort_index()
P=pd.concat(px,axis=1).sort_index(); R=P.pct_change()
# risk-adjusted recovery: rebound from trailing 20d low, scaled by trailing 40d volatility
F=(P/P.rolling(20).min()-1)/(R.rolling(40).std()+1e-8)
rows=[]
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(1,len(F)-h):
  x=F.iloc[i-1]; y=R.iloc[i:i+h].sum()
  z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.x.corr(z.y,method='spearman')); ns.append(len(z))
 a=np.asarray(vals,float)
 print('H',h,'IC %.5f ICIR %.5f hit %.3f dates %d avgN %.2f'%(np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0),len(a),np.mean(ns)))
 if h==10:
  for n in [180,500,750]:
   q=a[-min(n,len(a)):]; print('recent',n,'IC %.5f ICIR %.5f hit %.3f'%(np.nanmean(q),np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12),np.mean(q>0)))
print('coverage %.4f turnover %.4f instruments %d'%(F.notna().mean().mean(),F.rank(axis=1,pct=True).diff().abs().mean().mean(),len(px)))
out=F.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20340612_recovery_signal.csv',index=False)
print('artifact rows',len(out),'range',F.index.min(),F.index.max())
