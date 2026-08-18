import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-04-28')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}; p=pd.concat(D,axis=1).sort_index().loc[:cut]
lr=np.log(p).diff(); r10=lr.rolling(10).sum(); avg=r10.mean(axis=1); csvol=lr.rolling(20).std().mean(axis=1)
# Contrarian residual return, activated only when cross-asset dispersion is high.
disp=r10.std(axis=1); gate=disp>disp.rolling(60).quantile(.60)
f=(-(r10.sub(avg,axis=0))).div(lr.rolling(20).std().clip(.005,1),axis=0).where(gate).shift(1)
def run(h):
 R=np.log(p.shift(-h)/p); A=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],R.loc[dt]],axis=1).dropna()
  if len(z)>=8:A.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 a=pd.DataFrame(A,columns=['d','ic','n']).set_index('d');
 def s(x): return (x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean(),len(x),x.n.mean())
 print('H',h,'all',s(a),'365',s(a.loc[cut-pd.Timedelta(365,'d'):]),'730',s(a.loc[cut-pd.Timedelta(730,'d'):]))
 if h==10:print('coverage',f.notna().mean().mean(),'gate',gate.mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20]:run(h)
