import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3600)
 if x is not None:
  x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
# reversal after unusually weak recent performance, emphasized near lower part of medium-term range
ret5=P/P.shift(5)-1
lo=P.rolling(60,min_periods=40).min(); hi=P.rolling(60,min_periods=40).max()
loc=(P-lo)/(hi-lo).replace(0,np.nan)
f=(-ret5*(1-loc)).shift(1)
# 10-session forward return, no lookahead
fr=P.shift(-10)/P-1
rows=[]; sig=[]
for d in f.index:
 a=f.loc[d]; b=fr.loc[d]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  rows.append(z.iloc[:,0].corr(z.iloc[:,1]))
  sig.append(a.rank(pct=True))
ic=np.array(rows); print('dates',len(ic),'avg_instruments',np.mean([len(pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()) for d in f.index if len(pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna())>=8]))
print('mean_ic %.9f icir %.9f hit %.4f'%(np.nanmean(ic),np.nanmean(ic)/np.nanstd(ic,ddof=1),np.mean(ic>0)))
for n in [120,260,520,780,1200]:
 q=ic[-n:]; print('window',n,'ic %.9f icir %.9f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1)))
# rank turnover consecutive valid signals
sr=pd.DataFrame(sig); print('coverage %.4f turnover %.4f'%(P.notna().mean().mean(), np.nanmean(np.abs(sr.diff()).mean(axis=1))))
