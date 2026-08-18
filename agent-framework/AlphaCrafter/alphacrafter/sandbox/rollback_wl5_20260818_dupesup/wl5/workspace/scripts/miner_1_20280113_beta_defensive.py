import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-01-12'); b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:end].ffill(); R=P.pct_change(); m=R.median(axis=1)
# Defensive beta: negative 60d rolling beta to equal-weight benchmark, standardized by residual risk.
cov=R.rolling(60).cov(m); var=m.rolling(60).var(); beta=cov.div(var,axis=0); resid=R.sub(beta.mul(m,axis=0)); idv=resid.rolling(20).std(); f=-beta/(idv.clip(lower=1e-6))
y=P.shift(-10)/P-1
def calc(x,sl=slice(None)):
 a=[]; ns=[]
 for d in x.loc[sl].index:
  z=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('candidate dates avgN IC ICIR hit',calc(f));
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-12')]: print('REG',lo,hi,calc(f,slice(lo,hi)))
rk=f.rank(axis=1,pct=True);print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',(rk-rk.shift()).abs().mean(axis=1).dropna().mean());f.to_csv('scripts/miner_1_20280113_beta_defensive_signal.csv')
