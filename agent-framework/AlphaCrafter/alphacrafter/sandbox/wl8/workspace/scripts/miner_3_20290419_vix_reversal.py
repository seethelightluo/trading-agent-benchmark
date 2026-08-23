import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'] for a in assets},axis=1).sort_index()
P=P.loc[:'2029-04-18']
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].reindex(P.index).ffill()
ret=P.pct_change(); vz=(vix-vix.rolling(60,min_periods=40).mean())/vix.rolling(60,min_periods=40).std()
factor=(-ret.rolling(5).sum().shift(1)).mul((1+.75*vz.shift(1)).clip(.25,2.5),axis=0)
fwd=P.shift(-5)/P-1; rows=[]
for dt in P.index:
 z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def st(q):
 x=r.loc[q,'ic']; return len(x),r.loc[q,'n'].mean(),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()
print('period',r.index.min(),r.index.max(),'obs',len(r),'coverage',r.n.sum()/(len(r)*15))
for k,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent180',slice('2028-10-01','2029-04-18'))]: print(k,st(q))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]
 for dt in P.index:
  z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,len(x),x.mean(),x.mean()/x.std(ddof=1))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20290419_vix_reversal_signal.csv',index=False)
