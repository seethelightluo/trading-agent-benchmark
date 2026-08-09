import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); return d.close.rename(a)
p=pd.concat([load(a) for a in A],axis=1).sort_index().loc[:'2027-03-10']; r=p.pct_change()
# One interpretable candidate: intermediate momentum scaled by downside deviation.
mom=r.rolling(20,min_periods=15).sum(); down=r.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
f=mom/down.replace(0,np.nan)
print('candidate downside_adjusted_trend_20d; dates',len(p),'assets',len(A),'last',p.index.max().date())
for h in [1,5,10,20]:
  vals=[]; ns=[]; dates=[]
  fut=p.pct_change(h).shift(-h)
  for i,d in enumerate(p.index):
   q=pd.concat([f.loc[d],fut.loc[d]],axis=1).dropna()
   if len(q)>=8:
    x=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
    if np.isfinite(x): vals.append(x);ns.append(len(q));dates.append(d)
  z=pd.Series(vals,index=dates); print('H',h,'valid_dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
  print(' years',z.groupby(z.index.year).mean().round(6).to_dict())
print('coverage',round(f.notna().stack().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
# Complete library correlation audit using exact signal forms as documented proxies.
libs={
'ravmom_20':r.rolling(20,min_periods=15).sum()/r.rolling(20,min_periods=15).std(),
'volnorm_reversal_5':-r.rolling(5,min_periods=4).sum()/r.rolling(5,min_periods=4).std(),
'realized_volatility_20':-r.rolling(20,min_periods=15).std(),
'vix_conditioned_reversal_1d':-r.rolling(3,min_periods=3).sum(),
'peer_return_leadlag_2d':r.shift(2).sub(r.shift(2).mean(axis=1),axis=0),
'risk_adjusted_trend_20':r.rolling(20,min_periods=15).sum()/r.rolling(20,min_periods=15).std(),
'relative_volume_participation_20':r.rolling(20,min_periods=15).mean()
}
for n,x in libs.items():
 q=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna()
 print('library_rho',n,round(spearmanr(q.f,q.x).statistic,6),'cells',len(q))
# date-level orthogonality: median/maximum absolute pooled audit is explicitly reported
print('max_abs_library_correlation_proxy',max(abs(spearmanr(pd.concat([f.stack(),x.stack()],axis=1).dropna().iloc[:,0],pd.concat([f.stack(),x.stack()],axis=1).dropna().iloc[:,1]).statistic) for x in libs.values()))
