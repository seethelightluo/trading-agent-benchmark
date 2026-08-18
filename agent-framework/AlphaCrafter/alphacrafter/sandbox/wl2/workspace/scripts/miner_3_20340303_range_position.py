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
# Buy assets near the lower end of their 120d range, but only when the range is broad;
# lag one day to ensure no look-ahead.
lo=p.rolling(120).min(); hi=p.rolling(120).max(); pos=(p-lo)/(hi-lo+1e-12)
vol=r.rolling(20).std()/r.rolling(120).std().replace(0,np.nan)
f=(-pos.rank(axis=1,pct=True))*vol.rank(axis=1,pct=True); f=f.shift(1)
for h in [10,20,40]:
 fr=p.shift(-h)/p-1; rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index); q=z.loc['2026-07-16':'2034-03-02']
 print('horizon',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'coverage',f.loc[q.index].notna().mean().mean())
 for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-03-02'])]: print(lab,len(qq),qq.ic.mean(),qq.ic.mean()/qq.ic.std(ddof=1))
 if h==40:z.reset_index().to_csv('scripts/miner_3_20340303_range_position_ic.csv',index=False)
f.to_csv('scripts/miner_3_20340303_range_position_signal.csv')
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
