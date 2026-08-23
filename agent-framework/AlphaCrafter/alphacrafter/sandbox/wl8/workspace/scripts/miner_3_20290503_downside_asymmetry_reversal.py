import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'] for a in assets},axis=1).sort_index().loc[:'2029-05-02']
r=P.pct_change()
# Downside-asymmetry reversal: recent losses are emphasized when downside volatility dominates total volatility.
down=r.clip(upper=0).rolling(20,min_periods=15).std()
tot=r.rolling(20,min_periods=15).std()
asym=(down/tot).replace([np.inf,-np.inf],np.nan).clip(.25,2.0)
factor=(-r.rolling(5,min_periods=5).sum().shift(1))*asym.shift(1)
rows=[]
for dt in P.index:
 z=pd.concat([factor.loc[dt],(P.shift(-5)/P-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',r.index.min(),r.index.max(),'obs',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
for name,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent180',slice('2028-11-01','2029-05-02'))]:
 x=r.loc[q,'ic']; print(name,'obs',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]
 for dt in P.index:
  z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,'obs',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20290503_downside_asymmetry_signal.csv',index=False)
