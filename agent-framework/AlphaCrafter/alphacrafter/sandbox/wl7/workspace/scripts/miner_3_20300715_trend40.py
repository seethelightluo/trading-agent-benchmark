import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=3000)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').drop_duplicates('date');F[s]=d
rows=[]
for s,d in F.items():
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change(); v=r.rolling(40,min_periods=20).std()
 # medium-term trend excluding the most recent 5 sessions, volatility scaled and lagged
 sig=(c.pct_change(40).shift(5)/v).clip(-5,5).shift(1)
 rows.append(pd.DataFrame({'date':d.date,'asset':s,'signal':sig,'close':c}))
a=pd.concat(rows).sort_values(['date','asset'])
for H in [1,5,10,20]:
 a['fwd']=a.groupby('asset').close.shift(-H)/a.close-1; z=[]
 for dt,g in a.groupby('date'):
  g=g.dropna(subset=['signal','fwd'])
  if len(g)>=8:z.append((dt,len(g),g.signal.corr(g.fwd,method='spearman')))
 q=pd.DataFrame(z,columns=['date','n','ic']).dropna();m=q.ic.mean();ir=m/q.ic.std(ddof=1)*np.sqrt(252)
 print('H',H,'dates',len(q),'avg_n',q.n.mean(),'IC %.8f ICIR %.8f hit %.4f'%(m,ir,(q.ic>0).mean()))
 if H==1:q.to_csv('scripts/miner_3_20300715_trend40_ic.csv',index=False)
r=a.dropna(subset=['signal']).pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True)
print('coverage',a.signal.notna().groupby(a.date).mean().mean(),'turnover',(r.diff().abs().mean(axis=1)/2).dropna().mean())
a.to_csv('scripts/miner_3_20300715_trend40_signal.csv',index=False)
