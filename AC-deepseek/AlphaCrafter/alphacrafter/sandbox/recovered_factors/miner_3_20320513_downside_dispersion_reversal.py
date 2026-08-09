import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# downside-dispersion regime: broad negative median and unusually high cross-sectional downside spread
r5=P.pct_change(5); disp=r5.where(r5<0).std(axis=1); q=disp.rolling(120,min_periods=60).quantile(.75)
gate=(disp>q)&(r5.median(axis=1)<0)
# smooth reversal avoids one-day noise while retains short horizon
F=(-P.pct_change(3)).where(gate, np.nan)
print('data',P.index.min(),P.index.max(),'assets',len(assets),'gate days',int(gate.sum()),'coverage',F.notna().mean().mean())
def calc(h):
 fr=P.shift(-h)/P-1; vals=[]; ns=[]
 for d in P.index:
  x=F.loc[d]; y=fr.loc[d]; z=x.notna()&y.notna()
  if z.sum()>=8 and x.loc[z].nunique()>1 and y.loc[z].nunique()>1:
   vals.append(spearmanr(x.loc[z],y.loc[z]).statistic); ns.append(z.sum())
 s=pd.Series(vals); print(h,'dates',len(s),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
for h in [1,5,10,20]: calc(h)
# turnover across active dates
rank=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rank)):
 a,b=rank.iloc[i-1],rank.iloc[i]; z=a.notna()&b.notna()
 if z.sum()>=8: turns.append((a[z]-b[z]).abs().mean())
print('turnover10 proxy',np.mean(turns[::10]) if turns else np.nan)
# regimes
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31')]:
 sub=P.loc[lo:hi].index; vals=[]
 for d in sub:
  x=F.loc[d]; y=(P.shift(-1)/P-1).loc[d]; z=x.notna()&y.notna()
  if z.sum()>=8 and x[z].nunique()>1 and y[z].nunique()>1: vals.append(spearmanr(x[z],y[z]).statistic)
 s=pd.Series(vals); print('regime',lo,hi,len(s),s.mean(),s.mean()/s.std() if len(s)>1 else np.nan)
