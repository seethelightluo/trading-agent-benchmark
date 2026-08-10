import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2027-02-25'").set_index('date').close for s in S}; px=pd.DataFrame(P).sort_index(); r=px.pct_change(); ret10=px.pct_change(10); f=ret10*(1-r.rolling(5).std()/r.rolling(30).std())
out=[]
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; x=[]; ns=[]; cs=[]; dates=[]
 for d in px.index:
  ok=f.loc[d].notna()&fr.loc[d].notna()
  if ok.sum()>=8:
   x.append(spearmanr(f.loc[d,ok],fr.loc[d,ok]).statistic);ns.append(ok.sum());cs.append(ok.mean());dates.append(d)
 a=np.array(x);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'coverage',np.mean(cs))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
  z=[v for d,v in zip(dates,a) if lo<=d.year<=hi]
  if z: print(' ',lo, len(z),round(np.mean(z),5),round(np.mean(z)/np.std(z,ddof=1),4))
print('turn',f.rank(pct=True).diff().abs().mean().mean(),'matrixcov',f.notna().sum().sum()/f.size)
f.reset_index().to_csv('../persistent/factor_signals_miner_2_20270225_compressed_trend.csv',index=False)
