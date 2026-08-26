import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-12-22'); base='../persistent/stock_data'
px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
P=pd.DataFrame(px).sort_index().loc[:cut]; rets=np.log(P).diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
down=rets.clip(upper=0).rolling(20,min_periods=15).std()
raw=(-rets.rolling(10,min_periods=10).sum()/down.replace(0,np.nan)).shift(1)
stress=(vix > vix.rolling(60,min_periods=40).median()).astype(float)
factor=raw.mul(stress,axis=0).replace([np.inf,-np.inf],np.nan)
fwd=np.log(P.shift(-10)/P); rows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): rows.append((dt,len(z),ic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
def stat(x): return len(x),x.n.mean(),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean()
print('period',r.index.min(),r.index.max(),'dates',len(r),'assets',r.n.mean())
for name,x in [('full',r),('720d',r.tail(720)),('365d',r.tail(365)),('180d',r.tail(180))]: print(name,stat(x))
for h in [1,5,10,20]:
 yy=np.log(P.shift(-h)/P); q=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(q),len(q))
print('coverage_nonnull',factor.notna().sum().sum()/factor.size,'active_date_fraction',(stress>0).mean())
out='scripts/miner_2_20331222_stress_downside_pressure';r.to_csv(out+'_ic.csv');factor.to_csv(out+'_signal.csv')
