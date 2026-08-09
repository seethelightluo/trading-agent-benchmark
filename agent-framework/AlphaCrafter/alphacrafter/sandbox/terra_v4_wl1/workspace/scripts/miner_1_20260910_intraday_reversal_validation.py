import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-08-27')
D={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:cut]; D[s]=d
P=pd.DataFrame({s:D[s].close for s in U}).sort_index(); R=P.pct_change()
F=pd.DataFrame({s:-(D[s].close/D[s].open-1) for s in U}).sort_index().rolling(3,min_periods=2).mean()
# library-like signals
peer=pd.DataFrame({s: P.pct_change(5).drop(columns=s).median(axis=1) for s in U})
rev=-(P/P.shift(5)-1)
mom=P/P.shift(20)-1
for name,X in [('intraday',F),('peer',peer),('rev',rev),('mom',mom)]:
 print(name,'valid',X.stack().size)
print('correlations pooled')
for a,b in [('intraday','peer'),('intraday','rev'),('intraday','mom')]:
 X={'intraday':F,'peer':peer,'rev':rev,'mom':mom}; z=pd.concat([X[a].stack(),X[b].stack()],axis=1).dropna(); print(a,b,z.corr(method='spearman').iloc[0,1])
# date-level IC
for h in [1,5,10]:
 vals=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.corr(method='spearman').iloc[0,1])
 q=pd.Series(vals); print('H',h,'obs',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
print('coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
