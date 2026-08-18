import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): return None
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
ds={s:ld(s) for s in U}; close=pd.concat({s:d.close for s,d in ds.items()},axis=1); op=pd.concat({s:d.open for s,d in ds.items()},axis=1); hi=pd.concat({s:d.high for s,d in ds.items()},axis=1); lo=pd.concat({s:d.low for s,d in ds.items()},axis=1)
r=np.log(close).diff(); atr=((hi-lo)/close).rolling(20,min_periods=15).median()
# Fade unusually directional daily candles, normalized by recent range; cross-sectional demean removes common shock
raw=((np.log(close/op))/atr).replace([np.inf,-np.inf],np.nan)
f=(-(raw.sub(raw.median(axis=1),axis=0))).shift(1)
res=[]
for d in r.index:
 v=f.loc[d]; y=r.shift(-1).loc[d]; ok=v.notna()&y.notna()
 if ok.sum()>=8: res.append((d,spearmanr(v[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(res,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'assets',len(U),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-05-13')]:
 z=x.loc[a:b]; print(a,'dates',len(z),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
print('decay')
for h in [1,3,5,10]:
 y=r.rolling(h).sum().shift(-h); q=[]
 for d in r.index:
  v=f.loc[d]; yy=y.loc[d]; ok=v.notna()&yy.notna()
  if ok.sum()>=8:q.append(spearmanr(v[ok],yy[ok]).statistic)
 print(h,'dates',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1))
f.to_csv('scripts/miner_3_20330513_candle_reversal_signal.csv')
