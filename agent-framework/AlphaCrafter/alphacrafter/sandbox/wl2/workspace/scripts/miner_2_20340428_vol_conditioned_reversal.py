import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
# Volatility-conditioned short reversal: reverse recent 5-day move, attenuated for
# high current volatility relative to its 60-day baseline. Lagged one day.
rv20=r.rolling(20).std(); vr=rv20/(r.rolling(60).std()+1e-12)
f=(-p.pct_change(5)/(rv20*np.sqrt(5)+1e-12) * (1/(1+vr))).shift(1)
def ev(h):
 rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index); q=z.loc['2026-07-16':'2034-04-26']
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4),'coverage',round(f.loc[q.index].notna().mean().mean(),4))
 for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-04-26'])]: print(lab,len(qq),round(qq.ic.mean(),6),round(qq.ic.mean()/qq.ic.std(ddof=1),6))
 return z
for h in [10,20,40]:
 z=ev(h)
 if h==20:z.reset_index().to_csv('scripts/miner_2_20340428_vol_conditioned_reversal_ic.csv',index=False)
f.to_csv('scripts/miner_2_20340428_vol_conditioned_reversal_signal.csv'); print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
