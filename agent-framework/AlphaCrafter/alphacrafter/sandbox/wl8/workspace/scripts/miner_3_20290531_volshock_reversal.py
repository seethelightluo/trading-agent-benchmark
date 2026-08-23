import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-05-30']
r=P.pct_change(); v5=r.rolling(5,min_periods=5).std(); v40=r.rolling(40,min_periods=20).std(); shock=(v5/v40).shift(1).clip(.5,3)
# contrarian response to lagged 5d move, amplified when volatility has recently jumped
f=(-r.rolling(5,min_periods=5).sum().shift(1)*shock).clip(-5,5)
rows=[]
for d in P.index:
 z=pd.concat([f.loc[d],(P.shift(-5)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for nm,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-06-01','2029-05-30')),('recent180',slice('2028-12-01','2029-05-30'))]:
 x=R.loc[q,'ic']; print(nm,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; zlist=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:zlist.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(zlist); print('decay',h,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20290531_volshock_reversal_signal.csv',index=False)
