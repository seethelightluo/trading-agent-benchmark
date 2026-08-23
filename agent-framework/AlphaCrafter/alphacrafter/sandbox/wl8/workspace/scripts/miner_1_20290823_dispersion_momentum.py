import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-08-08']
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); mom=P.pct_change(20)
# Cross-sectional dispersion is observable at t and lagged one session for prediction.
disp=r.rolling(5,min_periods=5).std().mean(axis=1)
scale=(disp/disp.rolling(60,min_periods=30).median()).shift(1).clip(0.5,2.0)
# Momentum normalized by own volatility, with a common dispersion-regime intensity.
factor=(mom/vol).shift(1).mul(scale,axis=0)
rows=[]
for d in P.index:
 z=pd.concat([factor.loc[d],(P.shift(-10)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'dates',len(R),'avg_instruments',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for n,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-08-01','2029-08-08')),('recent180',slice('2029-02-01','2029-08-08'))]:
 x=R.loc[q,'ic']; print(n,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for h in [1,5,10,20]:
 vals=[]; fw=P.shift(-h)/P-1
 for d in P.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20290823_dispersion_momentum_signal.csv',index=False)
