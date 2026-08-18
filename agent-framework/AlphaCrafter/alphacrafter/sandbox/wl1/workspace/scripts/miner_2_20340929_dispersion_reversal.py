import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2034-09-29'
D={}
for s in U:
 try:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
  D[s]=d
 except Exception as e: print('missing',s,e)
px=pd.DataFrame(D).sort_index().ffill().loc[:cut]; r=np.log(px).diff()
down=r.clip(upper=0).rolling(20,min_periods=15).std(); disp=r.rolling(20,min_periods=15).std().mean(axis=1)
act=(disp/disp.rolling(120,min_periods=60).median()).clip(.5,2.0)
f=(-r.rolling(5,min_periods=5).sum()/down).mul(act,axis=0).shift(1)
for h in [5,10,20,40]:
 fr=np.log(px.shift(-h)/px); vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals); print('H',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'avgN',f.notna().sum(axis=1).mean(),'turn',(rank.diff().abs().mean(axis=1)/2).mean(),'range',f.index.min(),f.index.max())
fr=np.log(px.shift(-10)/px)
for a,b in [('2020','2024'),('2025','2029'),('2030','2034')]:
 qs=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(qs); print('REG',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
