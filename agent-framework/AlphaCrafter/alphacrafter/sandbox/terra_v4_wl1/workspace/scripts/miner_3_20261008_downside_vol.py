import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cut=pd.Timestamp('2026-10-08'); R={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 d=d[d.index<=cut]; R[s]=d.close.astype(float).pct_change()
R=pd.DataFrame(R); R.index=pd.to_datetime(R.index)
neg=R.where(R<0); f=-neg.rolling(20,min_periods=15).std()
for h in [1,3,5,10]:
 out=[]
 for dt in R.index:
  y=R.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt]
  q=pd.DataFrame({'f':f.loc[dt],'y':y}).dropna()
  if len(q)>=8: out.append((pd.Timestamp(dt),q.f.corr(q.y),len(q)))
 a=pd.DataFrame(out,columns=['dt','ic','n']); a['dt']=pd.to_datetime(a['dt']); x=a.ic
 print('H',h,'dates',len(x),'N',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
 if h==1:
  for yr,g in a.groupby(a.dt.dt.year): print('year',yr,round(g.ic.mean(),4),len(g))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('date_range',R.index.min(),R.index.max(),'valid_dates',f.notna().any(axis=1).sum())
