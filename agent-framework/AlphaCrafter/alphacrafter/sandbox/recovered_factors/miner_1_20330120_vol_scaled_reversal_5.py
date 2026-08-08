import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:'2033-01-19']
# Short-horizon cross-asset reversal: inverse lagged 5-day return, damped by 20d volatility.
r=p.pct_change(); v=r.rolling(20,min_periods=15).std(); f=(-p.pct_change(5)/(v+1e-10)).shift(1)
print('candidate=vol_scaled_reversal_5; dates',len(p),'assets',len(A),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 x=[];n=[]; fw=p.pct_change(h).shift(-h)
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 x=np.array(x);print('H',h,'dates',len(x),'meanN',round(np.mean(n),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for lo,hi in [('2024','2027-12-31'),('2028','2030-12-31'),('2031','2033-01-19')]:
 x=[];fw=p.pct_change(5).shift(-5)
 for d in f.loc[lo:hi].index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x); print('REG5',lo,hi,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4),'meanvalid',round(f.notna().sum(axis=1).replace(0,np.nan).mean(),2))
print('AUDIT_REQUIRED: no persistence without exact library signal correlation evidence')
