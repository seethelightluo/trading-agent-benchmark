import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}; r=pd.DataFrame({a:p[a].pct_change() for a in A}); m=r.median(axis=1); rel=r.sub(m,axis=0); beta=rel.rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var(),axis=0); resid3=rel.rolling(3,min_periods=3).sum()-beta.mul(m.rolling(3,min_periods=3).sum(),axis=0); raw=-resid3.div(rel.rolling(20,min_periods=10).std(),axis=0)
for h in [1,5,10]:
 rows=[]
 for dt in raw.index:
  vals=raw.loc[dt]; good=vals.dropna();
  if len(good)<8:continue
  med=good.median(); f=[];y=[]
  for a in A:
   if a not in vals or not np.isfinite(vals[a]) or dt not in p[a].index:continue
   i=p[a].index.get_loc(dt)
   if i+h<len(p[a]):f.append(vals[a]-med);y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8:rows.append(spearmanr(f,y).statistic)
 d=pd.Series(rows);print(h,len(d),d.mean(),d.mean()/d.std(),(d>0).mean())
