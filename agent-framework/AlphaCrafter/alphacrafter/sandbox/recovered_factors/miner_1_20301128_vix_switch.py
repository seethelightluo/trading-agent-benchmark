import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index(); r=p.pct_change(); mom=p/p.shift(20)-1; res=mom.sub(r.mean(axis=1).rolling(20).sum(),axis=0); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill(); vr=v.pct_change(20)
# macro state determines contrarian sign; stress (VIX rising) reversal, calm continuation
f=-res.where(vr>0,res*-1) # equivalent: stress -res, calm +res
# actually where condition false uses res*-1? fix calm momentum = res
f=-res.where(vr>0,res) # stress -res, calm -res? pandas unary precedence; explicit below
f=res.copy(); f[vr>0]=-res[vr>0]
y=p.shift(-10)/p-1
for name,x in [('vix_switch',f),('stress_reversal',-res.where(vr>0)),('calm_reversal',-res.where(vr<=0))]:
 q=[]
 for d in x.index:
  ok=x.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(x.loc[d,ok],y.loc[d,ok]).statistic)
 q=pd.Series(q).dropna(); print(name,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
 print('recent',q.tail(120).mean(),q.tail(120).mean()/q.tail(120).std(ddof=1))
print('coverage',f.notna().mean().mean(),'stress dates',(vr>0).sum())
