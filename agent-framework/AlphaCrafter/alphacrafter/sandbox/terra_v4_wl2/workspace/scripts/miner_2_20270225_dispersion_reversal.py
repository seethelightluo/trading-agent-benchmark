import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2027-02-25'").set_index('date').close for s in S}
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); disp=r.rolling(5).std().mean(axis=1); threshold=disp.rolling(60,min_periods=40).median(); active=disp.shift(1)>threshold.shift(1)
f=(-(px.pct_change(3))/r.rolling(20).std()).where(active, np.nan)
out=[]
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; x=[]; ns=[]; dates=[]; cov=[]
 for d in px.index:
  ok=f.loc[d].notna()&fr.loc[d].notna()
  if ok.sum()>=8:
   v=spearmanr(f.loc[d,ok],fr.loc[d,ok]).statistic
   if np.isfinite(v): x.append(v); ns.append(ok.sum()); cov.append(ok.mean()); dates.append(d)
 a=np.array(x); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2) if len(a) else 0,'IC',round(np.mean(a),5) if len(a) else None,'ICIR',round(np.mean(a)/np.std(a,ddof=1),5) if len(a)>1 else None,'hit',round(np.mean(a>0),4) if len(a) else None,'coverage',round(np.mean(cov),4) if len(a) else None)
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
  z=[v for d,v in zip(dates,a) if lo<=d.year<=hi]
  if len(z)>3: print(' ',lo,len(z),round(np.mean(z),5),round(np.mean(z)/np.std(z,ddof=1),4))
print('turn active',round(f.rank(pct=True).diff().abs().mean().mean(),4),'matrixcov',round(f.notna().sum().sum()/f.size,4),'active',round(active.mean(),3))
f.reset_index().to_csv('../persistent/factor_signals_miner_2_20270225_dispersion_reversal.csv',index=False)
