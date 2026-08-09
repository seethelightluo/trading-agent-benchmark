import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cut=pd.Timestamp('2026-12-17')
def load(s):
 d=pd.read_csv(base+'/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); return d.close.loc[:cut]
P=pd.concat({s:load(s) for s in U},axis=1).sort_index(); R=P.pct_change(fill_method=None)
# residual momentum: 20d return neutralized by trailing 60d beta to equal-weight benchmark, all lagged
bench=R.mean(axis=1); cov=R.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var()
beta=cov.div(var,axis=0); resid=R.sub(beta.mul(bench,axis=0)); f=resid.rolling(20,min_periods=15).sum().shift(1)
print('candidate residual momentum vs equal-weight benchmark')
for h in [1,5,10]:
 ic=[]; ns=[]
 for t in f.index:
  x=f.loc[t]; y=R.shift(-h).rolling(h).sum().loc[t]
  z=pd.concat([x,y],axis=1).dropna();
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=pd.Series(ic).dropna(); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('period',P.index.min(),P.index.max(),'symbols',len(U))
