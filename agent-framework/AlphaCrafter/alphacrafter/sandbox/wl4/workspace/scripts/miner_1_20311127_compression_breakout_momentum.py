import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2031-11-26'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close[d.index<=cutoff]
px=pd.DataFrame(px).sort_index(); ret=px.pct_change();
# Momentum is favored when recent volatility is compressed relative to its long baseline.
vol10=ret.rolling(10,min_periods=8).std(); vol60=ret.rolling(60,min_periods=40).std()
compression=(vol60/(vol10+1e-8)).clip(0.25,4.0)
raw=px.pct_change(20).div(vol20:=ret.rolling(20,min_periods=15).std()+1e-8)*compression
f=raw.shift(1); fr=px.shift(-10)/px-1
vals=[]; dates=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): vals.append(q); dates.append(dt); ns.append(len(z))
x=pd.Series(vals,index=dates)
print('factor compression_breakout_momentum_10d')
print('dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 y=x[x.index>=x.index.max()-pd.Timedelta(days=n)]; print('recent',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6))
print('decay')
for h in [5,10,20]:
 frh=px.shift(-h)/px-1; vv=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],frh.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vv.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(h, round(np.nanmean(vv),6), round(np.nanmean(vv)/np.nanstd(vv,ddof=1),6),len(vv))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(px),'instruments',len(U))
