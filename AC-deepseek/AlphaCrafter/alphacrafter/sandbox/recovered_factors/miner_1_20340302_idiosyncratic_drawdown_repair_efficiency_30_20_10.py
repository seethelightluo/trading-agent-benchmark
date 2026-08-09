import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
# One idea: idiosyncratic drawdown-repair efficiency. A high score means the recent
# 10-day residual rebound offsets a large preceding residual drawdown efficiently.
files=glob.glob('../persistent/stock_data/*.csv')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}
for s in watch:
 p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].replace(0,np.nan)
 cl[s]=d
px=pd.DataFrame(cl).sort_index().loc[:'2034-03-01']
r=px.pct_change(); med=r.median(axis=1)
# beta-neutralize each asset using completed 90-day trailing covariance
beta=r.rolling(90,min_periods=70).cov(med).unstack().reindex(columns=watch).div(med.rolling(90,min_periods=70).var(),axis=0)
res=r-beta.mul(med,axis=0)
# trailing residual path: a severe pre-window trough followed by a 10-day repair
# efficiency uses log-like additive residual returns and caps denominator only for numerical stability
path=res.rolling(30,min_periods=25).sum()
trough=path.shift(10).rolling(20,min_periods=15).min()
recent=res.rolling(10,min_periods=8).sum()
raw=recent/(path.shift(10)-trough).abs().clip(lower=.002)
# Require actual prior drawdown; otherwise cross-sectional neutral (not a fake extreme)
sig=raw.where((path.shift(10)-trough)<-.002)

def report(h):
 vals=[]; ns=[]
 fwd=px.shift(-h)/px-1
 for dt in sig.index:
  x=sig.loc[dt]; y=fwd.loc[dt]
  ok=x.notna()&y.notna()
  if ok.sum()>=8:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z): vals.append((dt,z));ns.append(ok.sum())
 a=np.array([z for _,z in vals]);
 print('H',h,'dates',len(a),'mean_names',round(np.mean(ns),3),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for name,lo in [('2020_2027','2020-01-01'),('2028_2030','2028-01-01'),('2031_now','2031-01-01'),('latest_6m','2033-09-01')]:
  q=np.array([z for d,z in vals if d>=pd.Timestamp(lo) and (name not in ['2020_2027'] or d<pd.Timestamp('2028-01-01')) and (name not in ['2028_2030'] or d<pd.Timestamp('2031-01-01'))])
  if len(q): print(' ',name,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
 return vals
allv={h:report(h) for h in [1,5,10,20]}
# coverage and turnover on daily cross-sectional ranks (neutral imputation excluded)
print('coverage',round(sig.notna().mean().mean(),6),'cells',int(sig.notna().sum().sum()),'of',sig.size)
ranks=sig.rank(axis=1,pct=True); print('rank_turnover',round(ranks.diff().abs().stack().mean(),6),'median_iqr',round(sig.quantile(.75,axis=1).sub(sig.quantile(.25,axis=1)).median(),6))
print('endpoint',px.index.max().date(),'rows',len(px),'assets',px.shape[1])
