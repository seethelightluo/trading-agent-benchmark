import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
vol=pd.DataFrame({s:d[s]['volume'] for s in U}); ret=pd.DataFrame({s:d[s]['close'].pct_change() for s in U})
# volume surprise, low surprise hypothesize liquidity mean reversion
f=-(vol/vol.rolling(20,min_periods=10).mean()-1); y=ret.shift(-1); I=[];N=[];D=[]
for dt in f.index:
 ok=f.loc[dt].notna()&y.loc[dt].notna()&np.isfinite(f.loc[dt])
 if ok.sum()>=8:
  I.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic);N.append(ok.sum());D.append(dt)
I=np.array(I); print('negative_volume_surprise_20d',len(I),np.mean(N),np.mean(I),np.mean(I)/np.std(I,ddof=1),np.mean(I>0),np.mean(N)/15)
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 z=I[(np.array(D,dtype='datetime64[ns]')>=np.datetime64(a+'-01-01'))&(np.array(D,dtype='datetime64[ns]')<=np.datetime64(b+'-12-31'))]; print(a+'-'+b,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
