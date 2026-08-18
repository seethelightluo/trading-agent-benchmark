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
mom=p.pct_change(60); vol=r.rolling(20,min_periods=10).std()*np.sqrt(252)
f=(-(mom/vol)).shift(1).rank(axis=1,pct=True)
rows=[]
for h in [5,10,20,40]:
 fr=p.shift(-h)/p-1
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,h,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [5,10,20,40]:
 q=z[z.h==h].set_index('date')
 ic=q.ic.mean(); print('H',h,'dates',len(q),'avgN',round(q.n.mean(),3),'IC',round(ic,6),'ICIR',round(ic/q.ic.std(ddof=1)*np.sqrt(252),6),'hit',round((q.ic>0).mean(),4))
 for label,qq in [('2026_2028',q.loc['2026':'2028']),('2029_2033',q.loc['2029':'2033-12-31']),('2031_2034',q.loc['2031':'2034-02-17'])]:
  x=qq.ic.mean(); print(' ',label,'dates',len(qq),'IC',round(x,6),'ICIR',round(x/qq.ic.std(ddof=1)*np.sqrt(252),6))
print('assets',len(raw),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.diff().abs().mean(axis=1).mean(),4))
z.to_csv('scripts/miner_2_20340217_risk_adjusted_trend_ic.csv',index=False); f.to_csv('scripts/miner_2_20340217_risk_adjusted_trend_signal.csv')
