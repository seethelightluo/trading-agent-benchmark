import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
# Volatility-conditioned risk-adjusted reversal: prior 20d loss/vol, strengthened
# when the instrument's trailing volatility is low relative to its own 120d history.
ret20=np.log(p/p.shift(20)); vol20=r.rolling(20).std();
volrank=vol20.rolling(120,min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],raw=False)
f=(-ret20/(vol20+1e-12))*(1-volrank)
f=f.shift(1)
rows=[]
for d in f.index:
 a=pd.concat([f.loc[d],(p.shift(-20)/p-1).loc[d]],axis=1).dropna()
 if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index); q=z.loc['2026-07-16':'2034-05-11']
def ir(x): return x.ic.mean()/x.ic.std(ddof=1) if len(x)>1 and x.ic.std(ddof=1)>0 else np.nan
print('dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',ir(q),'hit',(q.ic>0).mean(),'coverage',f.loc[q.index].notna().mean().mean())
for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-05-11'])]: print(lab,len(qq),qq.ic.mean(),ir(qq),(qq.ic>0).mean())
for h in [10,40]:
 rr=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8: rr.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print('decay',h,'IC',np.nanmean(rr),'n',len(rr))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_1_20340512_volconditioned_reversal_signal.csv'); z.reset_index().to_csv('scripts/miner_1_20340512_volconditioned_reversal_ic.csv',index=False)
