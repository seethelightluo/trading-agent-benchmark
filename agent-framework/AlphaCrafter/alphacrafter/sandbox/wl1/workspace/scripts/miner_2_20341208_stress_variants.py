import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def R(s,k='stock_data'): return pd.read_csv('../persistent/'+k+'/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
p=pd.concat({s:R(s) for s in U},axis=1).sort_index().ffill(); r=p.pct_change(); v=R('VIX','index_data').reindex(p.index).ffill(); m=r.rolling(20).sum(); q=v.pct_change(10).clip(-1,1)
for name,f in [('mom',m),('stressboost',m*(1+2*q).clip(-1.5,3)),('stressreverse',-m*(1+2*q).clip(-1.5,3)),('contrarian',-m)]:
 f=f.shift(1); a=[]
 for i in range(len(p)-10):
  z=pd.concat([f.iloc[i],r.iloc[i+1:i+11].sum()],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(a); print(name,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'latest',a[-1304:].mean(),a[-1304:].mean()/a[-1304:].std(ddof=1))
