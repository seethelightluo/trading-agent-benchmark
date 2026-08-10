import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
P=pd.DataFrame(p); r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); disp=r.std(axis=1).rolling(20,min_periods=15).mean(); threshold=disp.rolling(120,min_periods=60).quantile(.7)
# volatility-scaled 3d reversal, only elevated cross-asset dispersion; lag condition
f=(-r.rolling(3,min_periods=3).sum()/vol).where((disp>threshold).shift(1)); f=f.sub(f.median(axis=1),axis=0)
rows=[]; sig=[]
for dt in P.index:
 for a in A:sig.append((dt,a,f.loc[dt,a]))
 for h in [1,3,5]:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.concat([f.loc[dt].rename('f'),y.rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:rows.append((dt,h,spearmanr(q.f,q.y).statistic,len(q)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5]:
 q=d[d.h==h]; print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 for lo,hi in [('2025','2025-12'),('2026','2026-12'),('2027','2027-02-25')]:
  x=q.set_index('date').loc[lo:hi].ic; print(lo,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270227_disp_vol_reversal.csv',index=False)
print('coverage',round(out.signal.notna().mean(),4),'active_dates',f.notna().any(axis=1).sum())
