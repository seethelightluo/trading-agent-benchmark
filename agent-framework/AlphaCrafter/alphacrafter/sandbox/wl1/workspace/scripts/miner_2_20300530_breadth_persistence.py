import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy(); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:load(s) for s in U}).sort_index(); r=P.pct_change()
# Breadth-conditioned persistence: medium-term momentum rewarded only when the asset's
# daily return direction is persistently positive; normalize by realized volatility.
trend=P.pct_change(30)
breadth=(r>0).rolling(60,min_periods=40).mean()
vol=r.rolling(40,min_periods=25).std()
f=(trend/(vol+1e-8))*(0.5+breadth)
f=f.shift(1)
print('range',P.index.min(),P.index.max(),'assets',len(P.columns))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print(h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 for label,sub in [('2029+',q[q.index>='2029-01-01']),('2030YTD',q[q.index>='2030-01-01'])]:
  print(label,'dates',len(sub),'IC',round(sub.ic.mean(),6) if len(sub) else None,'ICIR',round(sub.ic.mean()/sub.ic.std(ddof=1),6) if len(sub)>1 else None)
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20300530_breadth_persistence_signal.csv',index=False)
print('coverage',round(out.symbol.nunique()/len(P.columns),4),'rows',len(out),'turnover',round(f.rank(pct=True).diff().abs().mean().mean(),6))
