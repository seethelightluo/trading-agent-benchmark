import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-03-31')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:cut]; lr=np.log(p).diff(); r10=lr.rolling(10).sum(); m=r10.mean(axis=1); disp=r10.std(axis=1); vol=lr.rolling(20).std()*np.sqrt(20)
raw=(-(r10.sub(m,axis=0))).div(vol.clip(lower=0.005,upper=1.0)).replace([np.inf,-np.inf],np.nan).clip(-5,5)
# Broader activation: dispersion above trailing 40th percentile, lagged one completed session.
f=raw.where(disp>disp.rolling(60).quantile(0.40)).shift(1); R=np.log(p.shift(-10)/p)
A=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],R.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:A.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(A,columns=['date','ic','n']).set_index('date')
def st(x):
 x=x.dropna(); return (x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean(),len(x),x.n.mean()) if len(x)>1 else (np.nan,)*5
print('cutoff',cut.date(),'dates/n',len(a),a.n.mean(),'min_n',a.n.min())
for k,x in [('all',a),('365',a.loc[cut-pd.Timedelta(days=365):]),('730',a.loc[cut-pd.Timedelta(days=730):]),('1095',a.loc[cut-pd.Timedelta(days=1095):]),('2028-30',a.loc['2028':'2030'])]:print(k,st(x))
print('coverage',f.notna().mean().mean(),'active_date_rate',f.notna().any(axis=1).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20]:
 rr=np.log(p.shift(-h)/p); q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('horizon',h,'ic',np.nanmean(q),'n',len(q))
