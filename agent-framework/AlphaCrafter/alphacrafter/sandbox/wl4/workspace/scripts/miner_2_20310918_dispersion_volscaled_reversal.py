import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-09-17'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); r10=P.pct_change(10); resid=r10.sub(r10.median(axis=1),axis=0)
vol=R.rolling(40,min_periods=20).std(); disp=R.std(axis=1); lagdisp=disp.shift(1); threshold=lagdisp.rolling(60,min_periods=30).median()
# lagged normalized residual reversal, active in compressed cross-asset regimes
F=(-resid/vol).shift(1).where(lagdisp<threshold,0.0)
print('rows',len(P),'period',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 fwd=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals); print(f'H{h} dates {len(a)} avgN {np.mean(ns):.2f} IC {np.nanmean(a):.6f} ICIR {np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(252/h):.6f} hit {np.mean(a>0):.3f}')
# recent windows H10
fwd=P.pct_change(10).shift(-10)
for n in [260,520,780]:
 vals=[]
 for dt in F.index[-n:]:
  z=pd.concat([F.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(vals); print('recent',n,'dates',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(252/10))
q=F.rank(pct=True); print('coverage',F.notna().mean().mean(),'turnover',q.diff().abs().mean().mean(),'active',float((lagdisp<threshold).mean()))
