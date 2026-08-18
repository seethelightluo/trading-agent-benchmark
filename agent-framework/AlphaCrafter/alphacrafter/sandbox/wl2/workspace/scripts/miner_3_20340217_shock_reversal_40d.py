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
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
short=r.rolling(5).sum(); vol=r.rolling(20).std()*np.sqrt(20)
shock=(short/(vol+1e-12)).clip(-4,4); trend=r.rolling(60).sum()/(r.abs().rolling(60).sum()+1e-12)
f=(-shock.rank(axis=1,pct=True))*((1-trend.abs()).clip(0,1).rank(axis=1,pct=True)); f=f.shift(1)
fr=p.shift(-40)/p-1; rows=[]
for d in f.index:
 a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index)
q=z.loc['2026-07-16':'2034-02-16']; ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
ranks=f.rank(axis=1,pct=True); common=ranks.notna().sum(axis=1)>=8
turn=ranks[common].diff().abs().mean(axis=1).mean()
print('factor=shock_reversal_40d assets',len(raw),'dates',len(q),'avgN',q.n.mean(),'IC',ic,'ICIR',ir,'hit',(q.ic>0).mean(),'coverage',f.loc[q.index].notna().mean().mean(),'turnover',turn)
for label,qq in [('2026_2028',q.loc['2026':'2028']),('2029_2033',q.loc['2029':'2033-12-31']),('2031_2034',q.loc['2031':'2034-02-16'])]: print(label,len(qq),qq.ic.mean(),qq.ic.mean()/qq.ic.std(ddof=1),(qq.ic>0).mean())
print('decay10',f.loc[q.index].corrwith(p.pct_change(10).shift(-10).loc[q.index],axis=1) if False else 'see prior script horizons')
z.reset_index().to_csv('scripts/miner_3_20340217_shock_reversal_40d_ic.csv',index=False); f.to_csv('scripts/miner_3_20340217_shock_reversal_40d_signal.csv')
