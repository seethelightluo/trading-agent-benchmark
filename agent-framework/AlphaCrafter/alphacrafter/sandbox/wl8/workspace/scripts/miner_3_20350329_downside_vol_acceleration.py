import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-03-29')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
px=pd.concat({a:load(a) for a in assets},axis=1).sort_index().loc[:END].ffill(); r=px.pct_change()
r20=px.pct_change(20); r60=px.pct_change(60)
accel=r20-r60/3
# downside realized risk, with floor to retain broad coverage
neg=r.where(r<0,0.0)
down=neg.rolling(30).std()*np.sqrt(252)
sig=(accel/(down+0.05*r.rolling(30).std()*np.sqrt(252))).shift(1)
rows=[]
for dt in px.index:
 x=sig.loc[dt]; y=px.shift(-10).loc[dt]/px.loc[dt]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): rows.append((dt,z,int(ok.sum())))
ic=np.array([q[1] for q in rows]); mean=ic.mean(); sd=ic.std(ddof=1)
print('period',rows[0][0].date(),rows[-1][0].date(),'dates',len(rows),'avgN',np.mean([q[2] for q in rows]))
print('IC10',mean,'ICIR_daily',mean/sd*np.sqrt(252),'hit',np.mean(ic>0),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).reindex([q[0] for q in rows]).mean(),'coverage',np.mean([q[2]/15 for q in rows]))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; out=[]
 for dt in px.index:
  x=sig.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: out.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.nanmean(out),len(out))
for n in [365,750,1260]:
 q=ic[-min(n,len(ic)):]; print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'dates',len(q))
out=pd.DataFrame({'date':sig.index}); [out.__setitem__(a,sig[a].values) for a in assets]; out.to_csv('scripts/miner_3_20350329_downside_vol_acceleration_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20350329_downside_vol_acceleration_ic.csv',index=False)
