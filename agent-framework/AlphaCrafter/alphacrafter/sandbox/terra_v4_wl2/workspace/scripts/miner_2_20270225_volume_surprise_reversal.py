import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in S:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2027-02-25'").set_index('date')
 D[s]=d
px=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); vol=pd.DataFrame({s:d.volume for s,d in D.items()}).reindex(px.index)
r=px.pct_change(); vratio=vol/vol.rolling(20,min_periods=10).median()-1
# volume-surprise confirmation: recent losers with unusually high volume tend to mean revert
f=-r.rolling(3).sum()*np.log1p(vratio.clip(lower=-.99))
fr={h:px.shift(-h)/px-1 for h in [1,3,5,10]}
for h in [1,3,5,10]:
 xs=[]; dates=[]; ns=[]; cov=[]
 for d in px.index:
  ok=f.loc[d].notna()&fr[h].loc[d].notna()
  if ok.sum()>=8:
   z=spearmanr(f.loc[d,ok],fr[h].loc[d,ok]).statistic
   if np.isfinite(z): xs.append(z);dates.append(d);ns.append(ok.sum());cov.append(ok.mean())
 a=np.array(xs); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(cov),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
  z=np.array([x for d,x in zip(dates,a) if lo<=d.year<=hi])
  if len(z)>1: print('REG',lo, len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('turnover',round(f.rank(pct=True).diff().abs().mean().mean(),5),'matrixcov',round(f.notna().sum().sum()/f.size,5))
f.reset_index().to_csv('../persistent/factor_signals_miner_2_20270225_volume_surprise_reversal.csv',index=False)
