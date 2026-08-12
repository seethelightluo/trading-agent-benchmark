import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150:d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.set_index('date').sort_index()
D={s:load(s) for s in U}; P=pd.DataFrame({s:d.close for s,d in D.items() if d is not None}).sort_index(); O=pd.DataFrame({s:d.open for s,d in D.items() if d is not None}).reindex(P.index)
# Prior-day intraday reversal, normalized by 20d volatility; all signals lagged one session
intr=O/P-1
vol=P.pct_change().rolling(20).std()+1e-8
f=(intr/vol).shift(1)
print('range',P.index.min(),P.index.max(),'assets',len(P.columns))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20300418_intraday_reversal_signal.csv',index=False)
print('coverage',out.symbol.nunique()/len(P.columns),'rows',len(out),'turnover',f.rank(pct=True).diff().abs().mean().mean())
