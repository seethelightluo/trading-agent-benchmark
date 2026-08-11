import pandas as pd,numpy as np
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for s in syms}
for look in [2,3,5,10]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); f=-r.rolling(look).sum()/r.rolling(20,min_periods=15).std(); y=x.close.shift(-1)/x.close-1
  z=pd.DataFrame({'f':f,'y':y}).dropna().reset_index(); rows.append(z)
 a=pd.concat(rows); v=[]
 for _,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:v.append(spearmanr(g.f,g.y).statistic)
 v=np.array(v); print(look,len(v),v.mean(),v.mean()/v.std(ddof=1),(v>0).mean())
