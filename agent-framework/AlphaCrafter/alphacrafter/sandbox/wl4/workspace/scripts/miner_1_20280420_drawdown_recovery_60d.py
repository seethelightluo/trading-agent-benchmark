import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),usecols=['date','close']);d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close
p=pd.DataFrame(P).sort_index();
# recovery strength: distance from trailing high, normalized by trailing volatility; contrarian recovery favors assets near highs after drawdown
r=p.pct_change(); high=p.rolling(60,min_periods=40).max().shift(1); dd=p.shift(1)/high-1; vol=r.rolling(20,min_periods=15).std().shift(1)*np.sqrt(20)
f=(dd/vol).replace([np.inf,-np.inf],np.nan); f=f.sub(f.median(axis=1),axis=0)
print('universe_dates=%d assets=%d'%(len(p),p.shape[1]))
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[dt][ok],fr.loc[dt][ok]).statistic
   if np.isfinite(q): z.append(q);ns.append(ok.sum())
 x=pd.Series(z);print('h=%d dates=%d avg_n=%.2f coverage=%.4f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(x),np.mean(ns),np.mean(ns)/15,x.mean(),x.mean()/x.std(),np.mean(x>0)))
 if len(x)>1:
  for n,y in [('early',x.iloc[:len(x)//2]),('late',x.iloc[len(x)//2:])]:print(' %s IC=%.6f ICIR=%.6f hit=%.4f'%(n,y.mean(),y.mean()/y.std(),np.mean(y>0)))
print('turnover=%.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
