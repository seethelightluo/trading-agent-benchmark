import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-05-10')
def ld(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
p=pd.concat({a:ld(a) for a in A},axis=1).sort_index().loc[:END].ffill(); r=p.pct_change(); b=r.mean(1); rr=r.sub(b,axis=0)
# Acceleration: recent 10-session relative trend minus preceding 30-session relative trend, normalized by 30d residual vol.
short=(1+rr).rolling(10).apply(np.prod,raw=True)-1; long=(1+rr).rolling(40).apply(np.prod,raw=True)-1
acc=short-long; v=rr.rolling(30).std()*np.sqrt(252); sig=acc/v.shift(1).replace(0,np.nan); sig=sig.shift(1)
rows=[]
for d in p.index:
 x=sig.loc[d]; y=p.shift(-10).loc[d]/p.loc[d]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): rows.append((d,z,int(ok.sum())))
ic=np.array([z for _,z,_ in rows]); m=ic.mean(); ir=m/ic.std(ddof=1)*np.sqrt(252); dates=[d for d,_,_ in rows]
turn=sig.rank(axis=1,pct=True).diff().abs().mean(1).reindex(dates).mean()
print('period',dates[0].date(),dates[-1].date(),'dates',len(rows),'avgN',np.mean([n for _,_,n in rows])); print('IC10',m,'ICIR_daily',ir,'hit',np.mean(ic>0),'turnover',turn,'coverage',np.mean([n/15 for _,_,n in rows]))
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;o=[]
 for d in p.index:
  x=sig.loc[d];y=fw.loc[d];ok=x.notna()&y.notna()
  if ok.sum()>=8:o.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.nanmean(o),len(o))
for n in [365,750,1260]:
 q=ic[-n:];print('recent',n,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),len(q))
out=pd.DataFrame({'date':sig.index});
for a in A:out[a]=sig[a].values
out.to_csv('scripts/miner_3_20350510_relative_trend_acceleration_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20350510_relative_trend_acceleration_ic.csv',index=False)
