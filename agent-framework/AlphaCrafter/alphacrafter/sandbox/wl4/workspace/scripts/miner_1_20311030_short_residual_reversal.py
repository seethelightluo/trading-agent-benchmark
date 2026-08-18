import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2031-10-15'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close[d.index<=cutoff]
px=pd.DataFrame(px).sort_index(); ret=px.pct_change(); r3=px.pct_change(3)
# Short-horizon residual shock reversal, robustly scaled by cross-sectional MAD; lag one completed day.
med=r3.median(axis=1); mad=r3.sub(med,axis=0).abs().median(axis=1)
f=(-(r3.sub(med,axis=0)).div(mad+1e-8,axis=0)).shift(1)
fr=px.shift(-10)/px-1; vals=[]; ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): vals.append(q); ns.append(len(z)); dates.append(dt)
x=pd.Series(vals,index=dates)
print('factor short_residual_reversal_3d'); print('dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 y=x[x.index>=x.index.max()-pd.Timedelta(days=9999)] if False else x.iloc[-n:]
 print('recent',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'price_dates',len(px),'instruments',len(U))
