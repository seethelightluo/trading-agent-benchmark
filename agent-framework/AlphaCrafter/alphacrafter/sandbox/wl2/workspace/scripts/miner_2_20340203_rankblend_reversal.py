import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>200:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=p.pct_change()
lo=p.rolling(60,min_periods=45).min(); hi=p.rolling(60,min_periods=45).max(); base=(-(2*p-lo-hi)/(hi-lo)).rank(axis=1,pct=True)
down=r.where(r<0).rolling(20,min_periods=5).std(); up=r.where(r>0).rolling(20,min_periods=5).std(); asym=((up-down)/(up+down)).rank(axis=1,pct=True)
# Equal rank blend: range reversal plus downside-volatility asymmetry preference, lagged.
f=(0.7*base+0.3*asym).shift(1); rows=[]; fr={h:p.shift(-h)/p-1 for h in [5,10,20]}
for d in f.index:
 for h in [5,10,20]:
  a=pd.concat([f.loc[d],fr[h].loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,h,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [5,10,20]:
 q=z[z.h==h].set_index('date')
 for label,qq in [('full',q),('2026_2028',q.loc['2026':'2028']),('2029_2033',q.loc['2029':'2033-12-31']),('2031_2034',q.loc['2031':'2034-02-03'])]:
  ic=qq.ic.mean(); print(h,label,'dates',len(qq),'avgN',round(qq.n.mean(),3),'IC',round(ic,6),'ICIR',round(ic/qq.ic.std(ddof=1),6),'hit',round((qq.ic>0).mean(),4))
print('assets',len(raw),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4)); z.to_csv('scripts/miner_2_20340203_rankblend_reversal_ic.csv',index=False); f.to_csv('scripts/miner_2_20340203_rankblend_reversal_signal.csv')
