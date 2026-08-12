import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy(); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:load(s) for s in U}).sort_index(); P=P.dropna(axis=1,how='all'); r=P.pct_change()
# Downside-resilient trend: medium-horizon return scaled by downside semideviation,
# with a positive slow trend gate; lagged one session.
down=r.clip(upper=0).pow(2).rolling(40,min_periods=25).mean().pow(.5)
f=(P.pct_change(20)/(down*np.sqrt(20)+1e-8)).where(P.pct_change(60)>0).shift(1)
print('range',P.index.min(),P.index.max(),'assets',len(P.columns))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); sd=q.ic.std(ddof=1)
 print('H',h,'dates',len(q),'avg_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/sd,'hit',(q.ic>0).mean())
 for lab,lo,hi in [('2020-25','2020','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-03-20')]:
  x=q.loc[lo:hi]
  if len(x)>1: print(' ',lab,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20300404_downside_resilient_signal.csv',index=False)
print('coverage',out.symbol.nunique()/len(P.columns),'rows',len(out),'turnover',f.rank(pct=True).diff().abs().mean().mean())
