import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-02-25'); P='../persistent/stock_data/'
C=pd.concat({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:END]
r=C.pct_change(); breadth=(r<0).sum(axis=1)/r.notna().sum(axis=1)
# Signal: cross-sectional contrarian 3d return, active only on historically extreme breadth.
raw=-(C.pct_change(3).shift(1).sub(C.pct_change(3).shift(1).median(axis=1),axis=0))
q=breadth.shift(1).rolling(252,min_periods=60).median(); active=breadth.shift(1)>=np.maximum(.60,q)
fwd=C.shift(-5)/C-1
rows=[]; ics=[]
for dt in C.index:
 if not active.get(dt,False): continue
 z=pd.concat([raw.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; ics.append(ic)
  for s in z.index: rows.append((dt,s,raw.loc[dt,s],int(len(z))))
a=np.asarray(ics); print('dates',len(a),'avgN',np.mean([x[3] for x in rows]) if rows else 0,'instruments',len(U),'IC %.9f ICIR %.9f hit %.6f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
 b=np.asarray([v for v,d in zip(a,[d for d in C.index if active.get(d,False) and pd.concat([raw.loc[d],fwd.loc[d]],axis=1).dropna().shape[0]>=8]) if lo<=d.year<=hi]); print('regime',lo,hi,'dates',len(b),'IC',b.mean() if len(b) else np.nan,'ICIR',b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
out=pd.DataFrame(rows,columns=['date','symbol','signal','n']); out.to_csv('../persistent/factor_signals_miner_1_20270225_adaptive_breadth.csv',index=False)
# rank turnover among active dates
if len(out):
 wide=out.pivot(index='date',columns='symbol',values='signal').rank(axis=1,pct=True); print('active rank turnover',wide.diff().abs().mean().mean())
print('coverage',out.symbol.nunique()/len(U),'period',C.index.min().date(),C.index.max().date())
