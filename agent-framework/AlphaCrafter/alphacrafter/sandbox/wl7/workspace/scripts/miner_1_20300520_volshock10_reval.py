import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150:d=get_index_daily_data(s,4000)
 if d is not None and len(d):p[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(p).sort_index().ffill(); ret=px.pct_change(); r10=px.pct_change(10); v20=ret.rolling(20).std(); v60=ret.rolling(60).std()
f=(-r10*(v20/v60)).shift(1); fw=px.shift(-10)/px-1
out=[]
for dt in px.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
for name,x in [('full',r),('early',r.iloc[:len(r)//3]),('mid',r.iloc[len(r)//3:2*len(r)//3]),('late',r.iloc[2*len(r)//3:])]:
 m=x.ic.mean(); sd=x.ic.std(ddof=1);print(name,'dates',len(x),'avg_n',x.n.mean(),'IC',m,'ICIR',m/sd*np.sqrt(252),'hit',(x.ic>0).mean())
for h in [1,5,10,20,40]:
 fw=px.shift(-h)/px-1; a=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a),len(a))
print('coverage',f.notna().sum().sum()/(f.shape[0]*15),'cutoff',px.index[-1])
r.to_csv('scripts/miner_1_20300520_volshock10_reval_ic.csv'); f.to_csv('scripts/miner_1_20300520_volshock10_reval_signal.csv')
