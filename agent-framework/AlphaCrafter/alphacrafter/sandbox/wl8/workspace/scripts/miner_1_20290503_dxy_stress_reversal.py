import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'] for a in assets},axis=1).sort_index().loc[:'2029-05-02']
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].reindex(P.index).ffill()
# DXY stress-conditioned reversal: recent asset weakness is favored, with stronger reversal under elevated dollar stress.
r=P.pct_change(); dz=(dxy-dxy.rolling(60,min_periods=40).mean())/dxy.rolling(60,min_periods=40).std()
factor=(-r.rolling(10,min_periods=10).sum().shift(1)).mul((1+0.75*dz.shift(1)).clip(.25,2.5),axis=0)
rows=[]
for dt in P.index:
 z=pd.concat([factor.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r0=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(x): return len(x),x['n'].mean(),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean()
print('period',r0.index.min(),r0.index.max(),'obs',len(r0),'coverage',r0.n.sum()/(len(r0)*15))
for k,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent180',slice('2028-10-15','2029-05-02'))]: print(k,stat(r0.loc[q]))
for h in [1,5,10,20]:
 vals=[]
 for dt in P.index:
  z=pd.concat([factor.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,len(x),x.mean(),x.mean()/x.std(ddof=1))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20290503_dxy_stress_reversal_signal.csv',index=False)
