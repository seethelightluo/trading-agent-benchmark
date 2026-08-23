import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'] for a in assets},axis=1).sort_index().loc[:'2029-06-27']
r=P.pct_change()
# Cross-asset dispersion reversal: fade each asset's recent return relative to the contemporaneous
# equal-weight universe move, scaled by its idiosyncratic volatility. All inputs lagged one day.
csmean=r.mean(axis=1)
relative=r.sub(csmean,axis=0)
idvol=relative.rolling(20,min_periods=15).std()
raw=-relative.rolling(5,min_periods=5).sum().shift(1)
factor=(raw/idvol.shift(1)).replace([np.inf,-np.inf],np.nan).clip(-5,5)
rows=[]
for dt in P.index:
 z=pd.concat([factor.loc[dt],(P.shift(-5)/P-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',R.n.mean(),'coverage',R.n.sum()/(len(R)*15))
for name,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-07-01','2029-06-27')),('recent180',slice('2028-12-01','2029-06-27'))]:
 x=R.loc[q,'ic']; print(name,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 vals=[]
 for dt in P.index:
  z=pd.concat([factor.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
out=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20290628_relative_dispersion_reversal_signal.csv',index=False)
