import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-06-13']
r=P.pct_change()
# Dispersion-conditioned residual reversal: reverse each asset's lagged 5-day return,
# amplify reversal in high cross-sectional dispersion and damp it in compressed markets.
resid=r.rolling(5,min_periods=5).sum().shift(1)
csmean=resid.mean(axis=1)
signal=-(resid.sub(csmean,axis=0))
disp=r.std(axis=1).rolling(20,min_periods=15).mean().shift(1)
rank=disp.rolling(120,min_periods=60).rank(pct=True).shift(1)
mult=(0.5+rank).clip(0.5,1.5)
factor=signal.mul(mult,axis=0)
fw=P.shift(-10)/P-1; rows=[]
for d in P.index:
 z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for name,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-06-14','2029-06-13')),('recent180',slice('2028-12-15','2029-06-13'))]:
 x=R.loc[q,'ic']; print(name,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]
 for d in P.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20290614_dispersion_reversal_signal.csv',index=False)
