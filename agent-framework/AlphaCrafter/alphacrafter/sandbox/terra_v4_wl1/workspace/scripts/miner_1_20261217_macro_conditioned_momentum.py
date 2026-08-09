import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
vix=pd.read_csv(Path('../persistent/index_data/VIX.csv'),parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# Macro-conditioned momentum: trend-following in calm/falling-VIX regimes, reversal in rising-VIX regimes.
vixchg=vix.pct_change(10)
raw=P.pct_change(5).div(R.rolling(20).std())
F=raw.mul(np.where(vixchg<0,1.0,-1.0),axis=0).shift(1)
# cross-sectional median demean, rank turnover
F=F.sub(F.median(axis=1),axis=0)
ics={h:[] for h in [1,5,10]}; dates={h:[] for h in ics}; nobs={h:[] for h in ics}
for i,d in enumerate(P.index):
  for h in ics:
    j=i+h
    if j>=len(P): continue
    x=F.iloc[i]; y=P.iloc[j]/P.iloc[i]-1
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
      ics[h].append(z.iloc[:,0].corr(z.iloc[:,1])); dates[h].append(d); nobs[h].append(len(z))
for h in ics:
 a=np.array(ics[h]); print(h,'dates',len(a),'avg_n',round(np.mean(nobs[h]),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round(np.mean(a>0),4))
# signal turnover among consecutive valid rows
rank=F.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(turn,6),'period',P.index.min().date(),P.index.max().date())
# regime split daily IC
for label,mask in [('vixfall',vixchg<0),('vixrise',vixchg>=0)]:
 a=[]
 for i,d in enumerate(P.index):
  if i+1>=len(P) or not mask.loc[d]: continue
  z=pd.concat([F.iloc[i],(P.iloc[i+1]/P.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print(label,'n',len(a),'IC',round(np.mean(a),8) if a else None)
