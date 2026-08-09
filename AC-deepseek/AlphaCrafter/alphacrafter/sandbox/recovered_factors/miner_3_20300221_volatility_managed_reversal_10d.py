import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index(); r=p.pct_change()
# Ten-day contrarian return, risk scaled by 20d realized volatility; lagged one full day.
f=-(p/p.shift(10)-1)/(r.rolling(20,min_periods=15).std()*np.sqrt(20)); f=f.shift(1)
# forward close-to-close returns, cross-sectional rank IC
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; z=[]; ns=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.asarray(z); print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
# ten-day rank turnover proxy and broad regime splits
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(10)).abs().stack().dropna().mean(); print('coverage',round(f.notna().stack().mean(),4),'turn10',round(turn,4),'rows',len(p),'assets',len(A))
y=p.shift(-10)/p-1
for name,st,en in [('2020-24','2020','2025'),('2025-27','2025','2028'),('2028-29','2028','2030'),('latest120',None,None)]:
 zz=[]
 ix=f.index[-120:] if name=='latest120' else f.loc[st:en].index
 for d in ix:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:zz.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic)
 print('REG',name,'dates',len(zz),'IC',round(np.mean(zz),5) if zz else None,'ICIR',round(np.mean(zz)/np.std(zz,ddof=1),5) if len(zz)>1 else None)
# complete signal-cell library audit against standard admitted base signals
lib={'risk_adjusted_trend_20':(p/p.shift(20)-1)/(r.rolling(20,min_periods=15).std()*np.sqrt(20)),'volnorm_reversal_5':-(p/p.shift(5)-1)/(r.rolling(5,min_periods=4).std()*np.sqrt(5)),'risk_adjusted_trend_20_raw':(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(),'short_reversal_5':-(p/p.shift(5)-1)}
mx=0; pair=''
for k,x in lib.items():
 q=pd.concat([f.stack().rename('f'),x.shift(1).stack().rename('x')],axis=1).dropna(); rho=spearmanr(q.f,q.x).statistic
 print('CORR',k,round(rho,5));
 if abs(rho)>mx:mx=abs(rho);pair=k
print('MAXCORR',pair,round(mx,5))