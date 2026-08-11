import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
c=pd.DataFrame({s:D[s].close for s in U}); r=c.pct_change(); v=r.rolling(20,min_periods=15).std()
# acceleration: short momentum minus fraction of medium momentum, volatility normalized, lagged
f=((c.pct_change(5)-c.pct_change(20)*.25)/(v*np.sqrt(5))).shift(1)
for h in [1,5,10]:
 y=c.shift(-h)/c-1;a=[];ns=[];tr=[];prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));q=f.loc[dt].rank(pct=True);tr.append(np.nan if prev is None else (q-prev).abs().mean());prev=q
 a=np.asarray(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),3),'turn',round(np.nanmean(tr),4))
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-07-12')]:
 a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],(c.shift(-1)/c-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.asarray(a);print('REG',lo,'dates',len(a),'IC',round(np.nanmean(a),6) if len(a) else 'nan','ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6) if len(a)>1 else 'nan')
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4))
