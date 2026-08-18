import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
base=Path('../persistent/stock_data'); assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv(base/f'{a}.csv'); d['date']=pd.to_datetime(d['date']); px[a]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); V=v.set_index('date')['close'].astype(float).reindex(P.index).ffill()
ret=P.pct_change(); r20=P/P.shift(20)-1; vol20=ret.rolling(20).std()*np.sqrt(252)
z=(V-V.rolling(120).median())/(V.rolling(120).std()+1e-12)
sig=r20.div(vol20+1e-12).mul(1+z.clip(-1.5,1.5)/2,axis=0).shift(1)
rows=[]
for i in range(len(P.index)-10):
 dt=P.index[i]; f=sig.iloc[i]; fr=P.iloc[i+10]/P.iloc[i]-1; ok=f.notna()&fr.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(f[ok],fr[ok]).statistic,ok.sum()))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',ic.n.sum()/(len(ic)*15))
print('IC10',ic.ic.mean(),'ICIR_daily_paper',ic.ic.mean()/(ic.ic.std(ddof=1)+1e-12)*np.sqrt(10),'hit',(ic.ic>0).mean())
for h in [5,10,20,40]:
 rr=[]
 for i in range(len(P.index)-h):
  f=sig.iloc[i]; fr=P.iloc[i+h]/P.iloc[i]-1; ok=f.notna()&fr.notna()
  if ok.sum()>=8: rr.append(spearmanr(f[ok],fr[ok]).statistic)
 print('decay',h,np.nanmean(rr))
for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 x=ic.loc[lo:hi].ic; print('regime',lo,hi,len(x),x.mean(),x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(10))
sig.to_csv('scripts/miner_2_20350427_vix_adaptive_momentum_signal.csv'); ic.to_csv('scripts/miner_2_20350427_vix_adaptive_momentum_ic.csv')
