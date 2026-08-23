import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-11-14']; r=P.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1); f=(-r.rolling(10,min_periods=8).sum().shift(1)/vol).replace([np.inf,-np.inf],np.nan)
def ev(h):
 fw=P.shift(-h)/P-1; rows=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [5,10,20]:
 q=ev(h); x=q.ic; print(h,len(x),round(q.n.mean(),3),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for nm,q in [('r360',ev(10).tail(360)),('r180',ev(10).tail(180))]: print(nm,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20291115_revalidate_volscaled_signal.csv',index=False)
