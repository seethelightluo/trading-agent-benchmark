import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end='2030-02-06'; d={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); x=x[x.date<=end].set_index('date').sort_index(); d[a]=x.close
px=pd.DataFrame(d).sort_index(); ret=px.pct_change()
sig=ret.rolling(60).sum()/ret.abs().rolling(60).sum(); s=sig.shift(1)
for h in [1,5,10,20]:
 fwd=px.pct_change(h).shift(-h); vals=[]; cells=0
 for date in s.index:
  z=pd.concat([s.loc[date],fwd.loc[date]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); cells+=len(z)
 v=np.array(vals); print('H',h,'dates',len(v),'meanN',cells/len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
r=sig.rank(axis=1,pct=True); print('coverage',sig.notna().sum().sum()/(sig.shape[0]*15),'turn10',r.diff(10).abs().mean().mean(),'dates',len(px))
for lo,hi in [('2020','2024'),('2025','2027'),('2028','2029'),('2029-10-01','2030-02-06')]:
 sub=[]; fwd=px.pct_change(10).shift(-10)
 for date in s.index:
  if str(date.date())>=lo and str(date.date())<=hi:
   z=pd.concat([s.loc[date],fwd.loc[date]],axis=1).dropna()
   if len(z)>=8: sub.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('REG',lo,hi,len(sub),np.mean(sub) if sub else np.nan,np.mean(sub)/np.std(sub,ddof=1) if len(sub)>1 else np.nan)
