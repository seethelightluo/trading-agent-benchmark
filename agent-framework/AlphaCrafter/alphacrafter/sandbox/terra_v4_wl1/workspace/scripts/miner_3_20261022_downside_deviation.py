import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-10-22')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); px[s]=d.loc[d.index<=cut,'close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(fill_method=None)
# robust downside deviation: zero contribution on up days, rolling mean with valid return count
neg2=R.clip(upper=0).pow(2)
count=R.notna().rolling(20).sum(); f=-np.sqrt(neg2.rolling(20,min_periods=10).sum()/count.clip(lower=10))
for h in [1,3,5,10]:
 z=[]
 for dt in R.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1
  q=pd.DataFrame({'f':f.loc[dt],'y':y}).dropna()
  if len(q)>=8:z.append((dt,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(z,columns=['date','ic','n']); x=a.ic
 print('H',h,'dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
 if h==1:
  for yr,g in a.groupby(a.date.dt.year):print('year',yr,'IC',round(g.ic.mean(),4),'n',len(g))
r=f.rank(axis=1,pct=True); print('coverage',f.notna().sum().sum()/f.size,'turnover',r.diff().abs().mean().mean(),'range',R.index.min(),R.index.max())
print('valid dates',f.notna().any(axis=1).sum())
