import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-04-14')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:cut]; lr=np.log(p).diff()
ret=lr.rolling(20).sum(); vol=lr.rolling(20).std().replace(0,np.nan)*np.sqrt(20)
eff=ret.abs()/lr.abs().rolling(20).sum().replace(0,np.nan)
# interpretable risk-adjusted trend quality, lagged to avoid lookahead
f=(ret/vol*eff.clip(0,1)).replace([np.inf,-np.inf],np.nan).clip(-10,10).shift(1)
R={h:np.log(p.shift(-h)/p) for h in [5,10,20]}
def obs(rr):
 A=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:A.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 return pd.DataFrame(A,columns=['date','ic','n']).set_index('date')
def stat(a):
 x=a.ic.dropna(); return dict(ic=x.mean(),icir=x.mean()/x.std(ddof=1),hit=(x>0).mean(),dates=len(x),avg_n=a.n.mean(),min_n=a.n.min())
print('cutoff',cut.date(),'rows',len(p),'instruments',p.shape[1])
for h,rr in R.items():
 a=obs(rr); print('horizon',h,'all',stat(a));
 for days in [365,730,1095]: print(' recent',days,stat(a.loc[cut-pd.Timedelta(days=days):]))
print('coverage',float(f.notna().mean().mean()),'active_date_rate',float(f.notna().any(axis=1).mean()),'turnover',float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
