import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2026-07-15'
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# macro-conditioned relative strength: 20d excess return, rewarded in low and falling VIX regimes
rel=P.pct_change(20).sub(P.pct_change(20).median(axis=1),axis=0)
vix_z=(vix-vix.rolling(120,min_periods=60).mean())/vix.rolling(120,min_periods=60).std()
# bounded regime multiplier, using only t information
reg=1/(1+np.exp(vix_z.clip(-4,4)))
f=rel.mul(reg,axis=0)
rows=[]
for h in [5,10,20]:
  ic=[]
  for dt in P.loc[:cut].index:
    if dt not in f.index: continue
    fut=P.shift(-h).loc[dt]/P.loc[dt]-1
    x=f.loc[dt]; z=pd.concat([x,fut],axis=1).dropna()
    if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.array(ic); rows.append((h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
print('dates',len(P.loc[:cut]),'instruments',len(U)); print('rows h n IC ICIR hit'); [print(*r) for r in rows]
# coverage and turnover at 10d observations
valid=f.loc[:cut].notna().sum(axis=1); print('coverage',valid.mean()/15,'avg_valid',valid.mean())
rank=f.loc[:cut].rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
for y,g in pd.Series([x[2] for x in []]).items(): pass
print('annual')
# redo annual 10d
ics=[]
for dt in P.loc[:cut].index:
 fut=P.shift(-10).loc[dt]/P.loc[dt]-1; z=pd.concat([f.loc[dt],fut],axis=1).dropna()
 if len(z)>=8: ics.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); print(d.groupby(d.index.year).ic.agg(['count','mean']))
# correlation to raw 20d momentum
raw=P.pct_change(20); print('raw corr',f.stack().corr(raw.stack()))
