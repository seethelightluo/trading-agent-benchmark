import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def d(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): return None
 z=pd.read_csv(p);z.date=pd.to_datetime(z.date);return z.sort_values('date')
f={}; y={}
for s in U:
 z=d(s)
 if z is not None:
  r=z.close.pct_change();f[s]=pd.Series((-r.rolling(20).std()).values,index=z.date);y[s]=pd.Series(z.close.pct_change().shift(-1).values,index=z.date)
x=pd.DataFrame(f).sort_index();q=pd.DataFrame(y).reindex(x.index);v=[];n=[];ds=[]
for dt in x.index:
 a=x.loc[dt];b=q.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8:v.append(a[ok].rank().corr(b[ok].rank()));n.append(ok.sum());ds.append(dt)
v=np.array(v);print('dates',len(v),'avg_n',np.mean(n),'coverage',np.mean(n)/15,'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15'),('2026-07-16','2027-02-25')]:
 m=(pd.DatetimeIndex(ds)>=a)&(pd.DatetimeIndex(ds)<=b);print(a,m.sum(),v[m].mean() if m.sum() else np.nan)
