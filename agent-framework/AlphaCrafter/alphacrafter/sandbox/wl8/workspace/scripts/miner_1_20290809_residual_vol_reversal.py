import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-08-08']
r=P.pct_change(); m=r.mean(axis=1)
# Volatility-scaled residual reversal: remove 40d rolling beta to equal-weight benchmark,
# then fade lagged 10d residual and divide by lagged 20d idiosyncratic volatility.
win=40
cov=r.rolling(win,min_periods=25).cov(m); var=m.rolling(win,min_periods=25).var()
beta=cov.div(var.replace(0,np.nan),axis=0)
res=(r.rolling(10,min_periods=8).sum()-beta.mul(m.rolling(10,min_periods=8).sum(),axis=0)).shift(1)
idio=(r-beta.mul(m,axis=0)).rolling(20,min_periods=12).std().shift(1)
factor=(-res/idio.replace(0,np.nan))
rows=[]
for d in P.index:
 for h in [10]:
  z=pd.concat([factor.loc[d],(P.shift(-h)/P-1).loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for n,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-08-08','2029-08-08')),('recent180',slice('2029-02-08','2029-08-08'))]:
 x=R.loc[q,'ic']; print(n,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [1,5,10,20]:
 vals=[]; fw=P.shift(-h)/P-1
 for d in P.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('decay',h,'obs',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20290809_residual_vol_reversal_signal.csv',index=False)
