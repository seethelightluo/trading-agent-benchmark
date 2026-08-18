import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-04-28')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:cut]
lr=np.log(p).diff(); r5=lr.rolling(5).sum(); avg=r5.mean(axis=1); vol=lr.rolling(20).std()*np.sqrt(20)
# Observation-only macro regime: VIX elevated relative to its trailing 60d distribution.
vix=pd.read_csv(Path('../persistent/index_data/VIX.csv'),parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill()
vz=(vix-vix.rolling(60).mean())/vix.rolling(60).std()
gate=(vz>0).astype(float)
raw=(-(r5.sub(avg,axis=0))).div(vol.clip(lower=.005,upper=1.0)).clip(-5,5)
f=raw.where(gate>0).shift(1)
A=[]
for h in [5,10,20]:
 R=np.log(p.shift(-h)/p); A=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],R.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:A.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 a=pd.DataFrame(A,columns=['date','ic','n']).set_index('date')
 def st(x):
  x=x.dropna(); return (x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean(),len(x),x.n.mean())
 print('H',h,'all/365/730',st(a),st(a.loc[cut-pd.Timedelta(days=365):]),st(a.loc[cut-pd.Timedelta(days=730):]))
 if h==10: print('dates/n',len(a),a.n.mean(),a.n.min(),'coverage',f.notna().mean().mean(),'active',gate.mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
