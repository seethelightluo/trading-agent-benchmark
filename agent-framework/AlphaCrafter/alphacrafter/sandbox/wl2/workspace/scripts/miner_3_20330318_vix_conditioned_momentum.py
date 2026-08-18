import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
r=pd.DataFrame(px).pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
# VIX-conditioned regime: low-vol continuation, high-vol shock reversal; all inputs lagged naturally
vixrank=vix.rolling(252,min_periods=126).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],raw=False)
mom=r.rolling(10,min_periods=10).sum()
sig=mom.where(vixrank<0.65,-mom).shift(1)
fwd=r.shift(-1)
ics=[]; dates=[]; cov=[]
for d in sig.index:
 x=sig.loc[d]; y=fwd.loc[d]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d); cov.append(len(z)/15)
ic=np.asarray(ics); print('dates',len(ic),'assets',15,'coverage',np.mean(cov),'IC',np.nanmean(ic),'ICIR',np.nanmean(ic)/np.nanstd(ic,ddof=1),'hit',np.mean(ic>0))
for h in [1,3,5,10]:
 yy=r.shift(-1).rolling(h).sum().shift(-(h-1)); a=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(a),len(a))
# rank turnover
rr=sig.rank(axis=1,pct=True); print('turnover',np.nanmean(np.abs(rr.diff()).sum(axis=1)/2))
for lo,hi in [(0,.35),(.35,.65),(.65,1)]:
 mask=(vixrank>=lo)&(vixrank<hi); aa=[]
 for d in sig.index:
  if not mask.get(d,False):continue
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',lo,hi,len(aa),np.nanmean(aa),np.nanmean(aa)/np.nanstd(aa,ddof=1) if len(aa)>1 else np.nan)
# artifact
out=sig.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_3_20330318_vix_conditioned_momentum_signal.csv')
print('artifact written')
