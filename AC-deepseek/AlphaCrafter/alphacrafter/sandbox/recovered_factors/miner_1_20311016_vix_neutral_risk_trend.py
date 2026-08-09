import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({Path(f).stem:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}).sort_index().ffill();px=px[[a for a in A if a in px]];r=px.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill(); vm=vix.pct_change(); w=80
cov=r.rolling(w,min_periods=50).apply(lambda x: 0,raw=True) # placeholder, overwritten below
# vectorized rolling covariance with macro
mr=vm.rolling(w,min_periods=50).mean(); vr=((vm-mr)**2).rolling(w,min_periods=50).mean()
beta=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
for c in px.columns:
 beta[c]=((r[c]*vm).rolling(w,min_periods=50).mean()-r[c].rolling(w,min_periods=50).mean()*mr)/vr
f=(px.pct_change(60)-beta*vix.pct_change(60))/(r.rolling(w,min_periods=50).std()*np.sqrt(252)+1e-8)
f=f.clip(lower=f.quantile(.05,axis=1),upper=f.quantile(.95,axis=1),axis=0)
print('idea=vix_beta_neutral_risk_adjusted_trend','instruments',len(f.columns),'rows',len(f),'coverage',round(f.notna().mean().mean(),4))
for h in [10,20,40]:
 fw=px.shift(-h)/px-1;z=[];ds=[];ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[d][ok],fw.loc[d][ok]).statistic
   if np.isfinite(q):z.append(q);ds.append(d);ns.append(ok.sum())
 z=pd.Series(z,index=ds);print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'latest120',round(z.tail(120).mean(),6),round(z.tail(120).mean()/z.tail(120).std(ddof=1),6))
print('turnover10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
