import pandas as pd,numpy as np
from scipy.stats import spearmanr
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in W}).sort_index().loc[:'2033-02-16']
# Breakout persistence: lagged distance of close from its 60-day low, scaled by 20-day range.
# Tests whether assets sustaining above their medium-term low continue outperforming.
lo=p.rolling(60,min_periods=40).min(); hi=p.rolling(60,min_periods=40).max()
sig=((p-lo)/(hi-lo+1e-12)).shift(1)
print('candidate breakout_position_60d; dates',len(p),'assets',len(W))
print('coverage',round(sig.notna().mean().mean(),4),'mean_valid',round(sig.notna().sum(axis=1).replace(0,np.nan).mean(),2),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for h in [1,5,10,20]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,lo_,hi_ in [('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-02-16')]:
 fw=p.pct_change(1).shift(-1); vals=[]
 for dt in sig.loc[lo_:hi_].index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(vals); print('REG1',label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('AUDIT_REQUIRED: exact library correlation audit required only if efficacy gates pass.')
