import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(D).sort_index().loc['2020-01-01':'2034-08-20'].ffill(); r=P.pct_change()
# relative trend: 20d asset return minus cross-sectional equal-weight return, normalized by idiosyncratic 40d vol; lagged
m=r.mean(axis=1); resid=r.sub(m,axis=0)
vol=resid.rolling(40,min_periods=25).std()*np.sqrt(20)
F=(P.pct_change(20).sub(P.pct_change(20).mean(axis=1),axis=0)/(vol+1e-12)).shift(1)
def ev(h,sub=F):
 R=P.shift(-h)/P-1; q=[]; ns=[]
 for d in sub.index:
  z=pd.concat([sub.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c): q.append(c);ns.append(len(z))
 q=np.array(q); return len(q),np.mean(ns),q.mean(),q.mean()/(q.std(ddof=1)+1e-12),(q>0).mean()
print('candidate relative_residual_momentum20_vol40 cutoff',P.index[-1].date(),'dates',len(P),'assets',len(A))
print('coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().stack().mean())
for h in [1,5,10,20]: print('H',h,ev(h))
for n in [180,500,750]: print('recent',n,ev(10,F.iloc[-n:]))
F.to_csv('scripts/miner_1_20340821_relative_resid_momentum20_signal.csv')
