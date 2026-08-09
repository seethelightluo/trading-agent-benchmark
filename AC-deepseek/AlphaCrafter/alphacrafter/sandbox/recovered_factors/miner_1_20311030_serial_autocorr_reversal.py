import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 a=f.rsplit('/',1)[-1][:-4]
 if a in A: D[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# Novel interpretable factor: negative lag-1 return autocorrelation, scaled by volatility.
# Positive values identify assets whose recent daily moves have mean-reverting serial dependence.
ac=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for a in A:
 x=r[a]; ac[a]=x.rolling(40,min_periods=25).apply(lambda z: pd.Series(z).autocorr(lag=1),raw=False)
f=-ac
print('dates',len(px),'assets',len(A),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z))
 s=np.array(vals)
 print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
