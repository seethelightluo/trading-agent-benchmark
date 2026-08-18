import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except Exception: pass
raw={s:fetch(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index()
p=p.loc[:'2034-03-16']; r=np.log(p).diff()
r20=r.rolling(20).sum(); r60=r.rolling(60).sum(); v=r.rolling(20).std()*np.sqrt(252)
# quality trend: medium trend risk-adjusted, confirmed by short trend; only positive trend earns score
f=(r60/(v+1e-8))*((r20>0).astype(float)+0.5*(r60>0).astype(float)); f=f.shift(1)
for h in [10,20,40]:
 fr=p.shift(-h)/p-1; rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.loc['2026-07-16':'2034-03-16']
 print('horizon',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'coverage',f.loc[q.index].notna().mean().mean())
 for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-03-16'])]: print(lab,len(qq),qq.ic.mean(),qq.ic.mean()/qq.ic.std(ddof=1) if len(qq)>1 else np.nan)
 if h==20: z.reset_index().to_csv('scripts/miner_1_20340317_trend_quality_ic.csv',index=False)
f.to_csv('scripts/miner_1_20340317_trend_quality_signal.csv')
print('assets',len(raw),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
