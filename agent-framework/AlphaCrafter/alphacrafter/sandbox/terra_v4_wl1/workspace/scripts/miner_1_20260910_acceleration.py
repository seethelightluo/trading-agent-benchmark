import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# acceleration: recent 5d return relative to its 20d average daily return
F= P.pct_change(5)-P.pct_change(20)/4
# rank-normalized not needed for IC
for h in [1,5,10]:
 vals=[]; dates=[]
 for dt in F.index:
  f=F.loc[dt]; y=P.shift(-h).loc[dt]/P.loc[dt]-1
  z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt)
 a=np.array(vals); print('H',h,'dates',len(a),'names_avg',round(np.mean([len(pd.concat([F.loc[d],(P.shift(-h).loc[d]/P.loc[d]-1)],axis=1).dropna()) for d in dates]),2),'IC',round(np.nanmean(a),5),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),5),'hit',round(np.mean(a>0),4))
# coverage/turnover ranks
valid=F.notna().sum(axis=1)/15
rank=F.rank(pct=True,axis=1)
turn=(rank-rank.shift(1)).abs().mean(axis=1).dropna().mean()
print('coverage',round(valid.mean(),4),'turnover',round(turn,4),'period',P.index.min(),P.index.max())
PY
python scripts/miner_1_20260910_acceleration.py