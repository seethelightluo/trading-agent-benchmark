import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in S}; px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# risk-adjusted medium-term trend, with cross-sectional volatility neutralization
f=px.pct_change(20)/r.rolling(20).std()
f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; x=[]; ns=[]; cs=[]
 for d in px.index:
  ok=f.loc[d].notna()&fr.loc[d].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[d,ok],fr.loc[d,ok]).statistic;x.append(q);ns.append(ok.sum());cs.append(ok.mean())
 x=np.array(x); print(h,len(x),np.mean(ns),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(cs))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
  z=[v for d,v in zip(px.index[-len(x):],x) if lo<=d.year<=hi]
  if z: print(lo,len(z),round(np.mean(z),5))
print('turn',f.rank(pct=True).diff().abs().mean().mean(),'matrixcov',f.notna().sum().sum()/f.size)
f.reset_index().to_csv('../persistent/factor_signals_miner_2_20270225_riskadj_trend20.csv',index=False)
