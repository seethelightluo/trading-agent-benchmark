import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close']
 px[s]=d
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# Candidate: volatility-normalized 60d momentum, conditionally invert in stressed/high breadth-failure regimes.
# factor at t uses only through t; forward starts t+1.
mom=P.pct_change(60); vol=r.rolling(40).std(); raw=mom/vol.replace(0,np.nan)
breadth=(r.rolling(20).sum()>0).mean(axis=1)
stress=(vix>vix.rolling(252,min_periods=60).median()*1.15)
# trend continuation in healthy breadth, reversal when stressed and breadth weak
f=raw.copy()
f.loc[(stress & (breadth<0.40)),:]*=-1
# mild cross-sectional de-meaning
f=f.sub(f.mean(axis=1),axis=0)
cut=pd.Timestamp('2032-04-29'); f=f.loc[:cut]; P=P.loc[:cut]
res=[]
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); res.append((h,len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
# turnover rank changes
rank=f.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).mean()
print('cutoff',cut.date(),'dates',len(f),'instruments',len(U),'coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',turn)
for x in res: print('H%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.3f'%x)
for third,g in enumerate(np.array_split(res and f.index,3),1):
 vals=[]; fw=P.shift(-10)/P-1
 for dt in g:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('third',third,'n',len(vals),'H10IC',np.mean(vals))
# signal artifact for provenance
out=f.stack().rename('signal').reset_index(); out.to_csv('scripts/miner_1_20320503_breadth_gated_momentum_signal.csv',index=False)
