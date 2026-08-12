import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy(); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()[['open','close','high','low']].astype(float)
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
O=pd.concat({s:x.open for s,x in D.items()},axis=1); C=pd.concat({s:x.close for s,x in D.items()},axis=1); H=pd.concat({s:x.high for s,x in D.items()},axis=1); L=pd.concat({s:x.low for s,x in D.items()},axis=1)
# Intraday weakness relative to recent intraday behavior, scaled by true range: short-term reversal with liquidity-independent normalization
intr=C/O-1
rng=(H-L)/C
f=-(intr-intr.rolling(20,min_periods=10).mean())/(rng.rolling(20,min_periods=10).median()+1e-6)
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
P=C
print('range',P.index.min(),P.index.max(),'assets',len(P.columns))
lastq=None
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); lastq=q
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 for start in ['2027-01-01','2029-01-01','2030-01-01']:
  x=q.loc[q.index>=start].ic
  print(start,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),len(x))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20300516_intraday_reversal_signal.csv',index=False)
print('coverage',out.symbol.nunique()/len(P.columns),'rows',len(out),'turnover',f.rank(pct=True).diff().abs().mean().mean())
