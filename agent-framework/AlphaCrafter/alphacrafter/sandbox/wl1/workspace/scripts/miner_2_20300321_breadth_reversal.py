import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150:d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({s:load(s) for s in U}).sort_index();r=P.pct_change(); r5=P.pct_change(5); r60=P.pct_change(60); v20=r.rolling(20,min_periods=15).std(); breadth=(r5>0).sum(axis=1)/r5.notna().sum(axis=1)
# Short-horizon reversal is only activated during broad weak breadth and scaled by risk;
# positive 60d trend avoids buying persistent structural losers, lagged one day.
f=(-r5/(v20+0.01)).where((breadth<0.45).values[:,None] & (r60>0)).shift(1)
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('H',h,'dates',len(q),'avg_n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
 for lab,lo,hi in [('2020-25','2020','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-03-20')]:
  x=q.loc[lo:hi];
  if len(x)>1:print(' ',lab,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20300321_breadth_reversal_signal.csv',index=False)
print('coverage',out.symbol.nunique()/len(P.columns),'rows',len(out),'turnover',f.rank(pct=True).diff().abs().mean().mean())
