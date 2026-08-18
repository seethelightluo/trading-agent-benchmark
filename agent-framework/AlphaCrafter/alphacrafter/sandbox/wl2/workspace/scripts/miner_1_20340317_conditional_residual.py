import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except Exception:pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index().loc[:'2034-03-16']; r=np.log(p).diff()
# Conditional residual reversal: reverse idiosyncratic 5d shocks only when market dispersion is elevated.
shock=r.rolling(5).sum(); med=shock.median(axis=1); resid=shock.sub(med,axis=0); vol=r.rolling(20).std(); disp=resid.std(axis=1); gate=(disp>disp.rolling(120).quantile(.65)).astype(float)
f=-(resid/(vol+1e-8))*gate.values[:,None]; f=f.shift(1)
for h in [10,20,40]:
 fr=p.shift(-h)/p-1; rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.loc['2026-07-16':'2034-03-16']
 print('horizon',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'coverage',f.loc[q.index].notna().mean().mean())
 for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-03-16'])]: print(lab,len(qq),qq.ic.mean(),qq.ic.mean()/qq.ic.std(ddof=1))
 if h==20:z.reset_index().to_csv('scripts/miner_1_20340317_conditional_residual_ic.csv',index=False)
f.to_csv('scripts/miner_1_20340317_conditional_residual_signal.csv'); print('assets',len(raw),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
