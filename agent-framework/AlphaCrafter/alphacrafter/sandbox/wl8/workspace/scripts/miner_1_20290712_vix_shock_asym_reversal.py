import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-07-11']
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# Volatility-shock asymmetric reversal: lagged 5d cross-sectional reversal is
# intensified after a positive VIX shock and mildly retained otherwise.
r=P.pct_change(5).shift(1)
shock=v.pct_change(3).shift(1)
cs=r.sub(r.mean(axis=1),axis=0)
# positive shock -> reversal; calm -> trend carry (interpretable asymmetric switch)
factor=cs.mul(np.where(shock>0.08,-1.0,0.35),axis=0)
rows=[]
for d in P.index:
 z=pd.concat([factor.loc[d],(P.shift(-10)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(x): return (len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean())
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for name,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-07-12','2029-07-11')),('recent180',slice('2029-01-13','2029-07-11'))]:
 x=R.loc[q,'ic']; n,ic,ir,hit=stat(x); print(name,'obs',n,'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]
 for d in P.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20290712_vix_shock_asym_reversal_signal.csv',index=False)
