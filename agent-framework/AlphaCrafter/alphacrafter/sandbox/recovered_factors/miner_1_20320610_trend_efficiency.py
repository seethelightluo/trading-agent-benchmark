import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 D[a]=d['close']
P=pd.DataFrame(D).sort_index(); R=P.pct_change()
# Trend efficiency: signed displacement divided by path length. Lagged one day.
# It rewards persistent directional movement and discounts choppy momentum.
path=R.abs().rolling(20,min_periods=15).sum()
F=(P.pct_change(20)/path).shift(1)
fr={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
print('data',P.index.min(),P.index.max(),'assets',len(assets),'dates',len(P),'coverage',F.notna().mean().mean())
def calc(h, idx=None):
 vals=[]; ns=[]
 for d in (P.index if idx is None else idx):
  x=F.loc[d]; y=fr[h].loc[d]; z=x.notna()&y.notna()&np.isfinite(x)&np.isfinite(y)
  if z.sum()>=8 and x[z].nunique()>1 and y[z].nunique()>1:
   vals.append(spearmanr(x[z],y[z]).statistic); ns.append(z.sum())
 s=pd.Series(vals); print('h',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4),'std',round(s.std(),5)); return s
for h in [1,5,10,20]: calc(h)
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-06-09')]:
 idx=P.loc[lo:hi].index; s=calc(1,idx)
# rank turnover every 10 decisions
rank=F.rank(axis=1,pct=True); ts=[]
for i in range(10,len(rank),10):
 z=rank.iloc[i-10].notna()&rank.iloc[i].notna()
 if z.sum()>=8: ts.append((rank.iloc[i-10][z]-rank.iloc[i][z]).abs().mean())
print('turnover10_proxy',np.mean(ts))
