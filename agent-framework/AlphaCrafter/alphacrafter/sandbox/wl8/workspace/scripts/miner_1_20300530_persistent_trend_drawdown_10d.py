import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-05-30')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}; p=pd.DataFrame({s:x[x.index<=cut] for s,x in px.items()}).sort_index(); r=p.pct_change()
# 30d trend with persistence and drawdown protection: reward persistent positive trend, penalize recent drawdown
r30=p.pct_change(30); up=r.rolling(30,min_periods=20).apply(lambda x:np.mean(x>0),raw=True); dd=p/p.rolling(60,min_periods=40).max()-1; vol=r.rolling(30,min_periods=20).std()
fac=(r30/vol)*(0.5+up)*(1+dd.clip(-.25,0))
ics=[]; dates=[]; ns=[]; cov=[]; turns=[]
for i in range(len(p)-10):
 dt=p.index[i]; end=p.index[i+10]
 if end>cut or dt<p.index[70]:continue
 x=fac.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 v=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(v):ics.append(v);dates.append(dt);ns.append(ok.sum())
 cov.append(ok.mean())
 if i:
  a=x.rank(pct=True);b=fac.iloc[i-1].rank(pct=True);o=a.notna()&b.notna()
  if o.sum():turns.append(abs(a[o]-b[o]).mean())
ics=np.array(ics);dates=np.array(dates,dtype='datetime64[ns]');m=ics.mean();print({'factor':'persistent_trend_drawdown_30d','dates':len(ics),'start':str(dates[0]),'end':str(dates[-1]),'avg_instruments':np.mean(ns),'coverage':np.mean(cov),'ic':m,'icir':m/ics.std(ddof=1),'hit':np.mean(ics>0),'turnover':np.mean(turns)})
for n,mask in [('recent180',dates>=np.datetime64('2029-12-01')),('recent360',dates>=np.datetime64('2029-01-01')),('2028',(dates>=np.datetime64('2028-01-01'))&(dates<np.datetime64('2029-01-01'))),('2029',(dates>=np.datetime64('2029-01-01'))&(dates<np.datetime64('2030-01-01'))),('2030',dates>=np.datetime64('2030-01-01'))]:
 z=ics[mask];print(n,len(z),z.mean() if len(z) else None,z.mean()/z.std(ddof=1) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':ics}).to_csv('scripts/miner_1_20300530_persistent_trend_drawdown_10d_ic.csv',index=False)
fac.iloc[-1].rename('signal').to_csv('scripts/miner_1_20300530_persistent_trend_drawdown_10d_signal.csv')
