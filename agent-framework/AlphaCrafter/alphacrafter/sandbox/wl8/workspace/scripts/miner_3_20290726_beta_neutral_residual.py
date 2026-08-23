import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:'2029-07-25']
r=P.pct_change(); m=r.mean(axis=1)
# Candidate: beta-neutral residual reversal. Remove each asset's rolling 60d beta
# to the equal-weight benchmark from its lagged 10d return, then fade the residual.
win=60
cov=r.rolling(win,min_periods=40).cov(m)
var=m.rolling(win,min_periods=40).var()
beta=cov.div(var,axis=0)
resid10=r.rolling(10,min_periods=10).sum().sub(beta.mul(m.rolling(10,min_periods=10).sum(),axis=0))
factor=(-resid10).shift(1)
rows=[]
for d in P.index:
 z=pd.concat([factor.loc[d],(P.shift(-5)/P-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('period',R.index.min(),R.index.max(),'obs',len(R),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.sum()/(len(R)*15),4))
for n,q in [('full',slice(None)),('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent360',slice('2028-07-01','2029-07-25')),('recent180',slice('2029-01-01','2029-07-25'))]:
 x=R.loc[q,'ic']; print(n,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4))
for h in [1,5,10,20]:
 x=[]; fw=P.shift(-h)/P-1
 for d in P.index:
  z=pd.concat([factor.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(x);print('decay',h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20290726_beta_neutral_residual_signal.csv',index=False)
