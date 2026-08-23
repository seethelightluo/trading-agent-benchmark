import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-03-04')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();P=P[P.index<=cut]; r=P.pct_change()
variants={'accel_60_20':P.pct_change(60)-P.pct_change(20),'smooth_mom_60':P.pct_change(60)/(r.rolling(20).std()*np.sqrt(20)+1e-8),'riskadj_60':P.pct_change(60)/(r.where(r<0,0).pow(2).rolling(60).mean().pow(.5)+1e-8)}
for nm,f in variants.items():
 y=P.shift(-10)/P-1; a=[]; cov=[]
 for d in f.index:
  ok=f.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:a.append(spearmanr(f.loc[d][ok],y.loc[d][ok]).statistic);cov.append(ok.sum()/15)
 a=np.array(a); print(nm,'dates',len(a),'n=15','IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)+1e-12),'hit',np.mean(a>0),'cov',np.mean(cov))
 for lo,hi in [(0,700),(700,1400),(1400,9999)]:
  z=a[lo:hi];print(' regime',len(z),z.mean(),z.mean()/(z.std(ddof=1)+1e-12))
