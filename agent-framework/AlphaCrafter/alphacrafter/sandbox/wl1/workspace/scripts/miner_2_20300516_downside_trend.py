import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy(); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:load(s) for s in U}).sort_index(); r=P.pct_change()
# Reward persistent trend, penalizing downside variation more than upside variation.
down=r.where(r<0,0).rolling(40,min_periods=20).std(); up=r.where(r>0,0).rolling(40,min_periods=20).std()
f=(P.pct_change(30)/(down+up+1e-8))*(1+0.5*(up/(down+1e-8)-1).clip(-1,1)); f=f.shift(1)
print('range',P.index.min(),P.index.max(),'assets',len(P.columns))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print(h,len(q),round(q.n.mean(),2),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6),round((q.ic>0).mean(),4))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20300516_downside_trend_signal.csv',index=False)
print('coverage',out.symbol.nunique()/len(P.columns),'rows',len(out),'turnover',f.rank(pct=True).diff().abs().mean().mean())
