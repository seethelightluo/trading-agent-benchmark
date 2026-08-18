import pandas as pd,numpy as np
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2034-10-26')
C={}; R={}
for s in SYMS:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); x=x.loc[x.index<=END]; C[s]=x.close; R[s]=x.close.pct_change()
C=pd.DataFrame(C); R=pd.DataFrame(R)
# breadth and cross-sectional residual reversal, all lagged
breadth=(R.rolling(20).sum()>0).mean(axis=1).shift(1)
med=R.rolling(5).sum().median(axis=1)
res=R.rolling(5).sum().subtract(med,axis=0)
vol=R.rolling(20,min_periods=15).std()*np.sqrt(252)
F=(-res/vol).mul((1+(0.8*(breadth<0.45).astype(float))),axis=0).shift(1)
# artifact
F.index.name='date'; F.loc[F.notna().sum(axis=1)>=8].reset_index().to_csv('../persistent/miner_1_20341027_stress_reversal_signal.csv',index=False)
for h in [1,5,10,20,40]:
 fr=C.shift(-h)/C-1; vals=[]; dates=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dates.append(dt);ns.append(len(z))
 a=np.array(vals);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
 for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2032-12-31'),('2033','2034-10-26')]:
  q=(pd.Series(dates)>=pd.Timestamp(lo))&(pd.Series(dates)<=pd.Timestamp(hi)); aa=a[q.values];print(' ',lo,len(aa),round(np.nanmean(aa)/np.nanstd(aa,ddof=1),5) if len(aa)>1 else None)
print('coverage',round(F.notna().sum().sum()/(len(F)*15),5),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).median(),5))
