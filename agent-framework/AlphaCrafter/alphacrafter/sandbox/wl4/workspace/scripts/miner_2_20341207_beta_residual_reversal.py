import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-12-06')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in px.values()])); P=pd.DataFrame({s:px[s].reindex(dates) for s in U}); r=P.pct_change(); m=r.mean(axis=1)
# rolling beta computed asset-by-asset, using only information through t, then lagged in signal
vm=m.rolling(60,min_periods=40).var(); beta=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U: beta[s]=r[s].rolling(60,min_periods=40).cov(m)/vm
res=r-beta.mul(m,axis=0); f=-res.rolling(20,min_periods=15).sum().shift(1)
def run(H,lo=0):
 vals=[]; ns=[]
 for i in range(lo,len(P)-H-1):
  x=f.iloc[i]; y=P.iloc[i+1+H]/P.iloc[i+1]-1; ok=x.notna()&y.notna(); ns.append(int(ok.sum()))
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic)
 a=np.array(vals); return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns),f.notna().mean().mean()
print('cutoff',cut.date(),'dates',len(P),'assets',len(U))
for h in [1,5,10,20]: print('H',h,run(h))
for n in [120,260,520,780]: print('recent',n,run(10,max(0,len(P)-n-11)))
print('turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>.1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_2_20341207_beta_residual_reversal_signal.csv',index=False)
