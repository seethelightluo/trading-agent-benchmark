import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>120:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].reindex(P.index).ffill()
med=v.rolling(252,min_periods=120).median().shift(1); high=(v.shift(1)>med).astype(float)
ret20=P.pct_change(20); vol40=R.rolling(40,min_periods=30).std(); base=ret20/(np.sqrt(20)*vol40)
sig=base.mul(1-2*high,axis=0).shift(1); sig=sig.sub(sig.mean(axis=1),axis=0)
rows=[]; signals=[]
for dt in P.index:
 x=sig.loc[dt]; y=R.shift(-1).loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].corr(y[ok]),int(ok.sum())))
 for s in U: signals.append((dt,s,sig.loc[dt,s]))
ic=pd.DataFrame(rows,columns=['date','ic','n']).dropna(); pd.DataFrame(signals,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20311103_vix_cond20_signal.csv',index=False); ic.to_csv('scripts/miner_2_20311103_vix_cond20_ic.csv',index=False)
for h in [1,5,10,20]:
 f=P.pct_change(h).shift(-h); z=[]
 for dt in P.index:
  a=sig.loc[dt]; b=f.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(a[ok].corr(b[ok]))
 z=pd.Series(z).dropna(); print(h,len(z),round(z.mean(),6),round(z.mean()/z.std(),6),round((z>0).mean(),4))
print('dates',len(ic),'avgN',round(ic.n.mean(),2),'coverage',round(sig.notna().sum().sum()/(len(sig)*len(U)),4),'turnover',round(((sig.rank(axis=1)-sig.shift(1).rank(axis=1)).abs().sum(axis=1)/sig.notna().sum(axis=1)).mean(),4))
for i,q in enumerate(np.array_split(np.arange(len(ic)),3)): print('third',i,round(ic.iloc[q]['ic'].mean(),6),len(q))
