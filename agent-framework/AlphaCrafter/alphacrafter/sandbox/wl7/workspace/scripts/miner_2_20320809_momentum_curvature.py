import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,days=2600)
  except Exception:pass
  if d is not None:break
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);frames[s]=d.sort_values('date').set_index('date')
close=pd.DataFrame({s:d.close for s,d in frames.items()}).sort_index(); lr=np.log(close).diff(); v40=lr.rolling(40,min_periods=25).std()*np.sqrt(252)
r5=np.log(close/close.shift(5));r20=np.log(close/close.shift(20))
# Momentum curvature: recent 5d trend relative to preceding 15d trend, volatility normalized.
f=((r5-r20)/v40).sub(((r5-r20)/v40).median(axis=1),axis=0)
print('universe',len(frames),'dates',len(f),'assets',close.shape[1],'coverage',round(f.stack().notna().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [1,5,10,20]:
 fw=np.log(close.shift(-h)/close);rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:rows.append((dt,z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),len(z)))
 o=pd.DataFrame(rows,columns=['date','ic','n']);ic=o.ic.mean();ir=ic/o.ic.std(ddof=1)
 print('H',h,'dates',len(o),'avgN',round(o.n.mean(),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((o.ic>0).mean(),4),'thirds',[round(x,6) for x in [o.iloc[:len(o)//3].ic.mean(),o.iloc[len(o)//3:2*len(o)//3].ic.mean(),o.iloc[2*len(o)//3:].ic.mean()]])
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20320809_momentum_curvature_signal.csv',index=False);print('artifact written')
