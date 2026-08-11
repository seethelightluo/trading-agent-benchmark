import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2026-12-16'
def load(s):
 return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.loc[:cut]
p=pd.concat({s:load(s) for s in U},axis=1).sort_index().ffill(); r=p.pct_change(); eq=r[U[:8]].mean(axis=1)
# Candidate: rolling equity-beta residual return, volatility-normalized.
# Residualizes each asset's 40d return against contemporaneous equity-basket return,
# then rewards idiosyncratic strength per unit total volatility.
rollcov=r.rolling(40,min_periods=25).cov(eq)
rolleq=eq.rolling(40,min_periods=25).var()
beta=rollcov.div(rolleq,axis=0)
resid=r.sub(beta.mul(eq,axis=0),axis=0)
f=(resid.rolling(20,min_periods=15).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-8)).shift(1)
fr=p.pct_change(10).shift(-10)
ics=[]; ns=[]; ds=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(d)
a=np.asarray(ics); print('candidate residual_beta_volscaled_10d'); print('cutoff',cut,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4)); print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
for y,g in pd.Series(a,index=ds).groupby(pd.DatetimeIndex(ds).year): print('year',y,'IC',round(g.mean(),6),'n',len(g))
for h in [5,20]:
 ff=(resid.rolling(20,min_periods=15).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-8)).shift(1); yy=p.pct_change(h).shift(-h); q=[]
 for d in ff.index:
  z=pd.concat([ff.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',round(np.mean(q),6),'ICIR',round(np.mean(q)/np.std(q,ddof=1),6),'dates',len(q))
