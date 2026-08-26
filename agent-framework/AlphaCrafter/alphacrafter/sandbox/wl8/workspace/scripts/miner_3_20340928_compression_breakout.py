import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date'); px[a]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# lagged volatility-compression breakout: medium return normalized by vol, amplified when recent vol is compressed vs long vol
mom=P.shift(1).pct_change(20); v20=r.shift(1).rolling(20).std(); v80=r.shift(1).rolling(80).std()
factor=(mom/v20)*(v80/v20).clip(0.5,2.5)
# winsorize/rank cross-section
sig=factor.rank(axis=1,pct=True)-.5
fwd=P.shift(-10)/P-1
rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turnover',sig.diff().abs().mean().mean())
for w in [365,750,1260]:
 z=q.tail(w); print('recent',w,z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr))
# signal artifact for audit
out=sig.tail(500); out.to_csv('scripts/miner_3_20340928_compression_breakout_signal.csv')
