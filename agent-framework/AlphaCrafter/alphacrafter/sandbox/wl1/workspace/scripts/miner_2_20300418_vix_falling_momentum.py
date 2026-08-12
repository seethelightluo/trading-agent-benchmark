import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150:d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy(); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:load(s) for s in U}).sort_index().dropna(axis=1,how='all')
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.reindex(P.index).ffill()
# Smooth macro condition: reward 20d trend when VIX is falling over 10d and below its 120d mean; otherwise retain a small trend signal
r20=P.pct_change(20); vol=P.pct_change().rolling(40).std()*np.sqrt(252)
macro=((v.diff(10)<0)&(v<v.rolling(120,min_periods=80).mean())).astype(float)
f=(r20/(vol+1e-8))*(0.35+0.65*macro.to_numpy()[:,None]); f=f.shift(1)
print('range',P.index.min(),P.index.max(),'assets',len(P.columns))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20300418_vix_falling_momentum_signal.csv',index=False)
print('coverage',out.symbol.nunique()/len(P.columns),'rows',len(out),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for label,q in [('full',q)]:
 print(label,'recent',q.loc[q.index>='2028-01-01'].ic.mean(),q.loc[q.index>='2028-01-01'].ic.mean()/q.loc[q.index>='2028-01-01'].ic.std(ddof=1))
