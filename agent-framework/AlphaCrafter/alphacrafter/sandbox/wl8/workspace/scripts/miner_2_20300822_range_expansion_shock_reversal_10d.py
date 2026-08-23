import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-08-21')
P={};H={};L={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); d=d[d.index<=cut]; P[s]=d.close;H[s]=d.high;L[s]=d.low
p=pd.DataFrame(P).sort_index(); h=pd.DataFrame(H).reindex(p.index); l=pd.DataFrame(L).reindex(p.index); prev=p.shift(1)
tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=0).groupby(level=0).max().div(p)
short=tr.rolling(20,min_periods=15).mean(); long=tr.rolling(60,min_periods=40).mean()
# Fade recent moves preferentially when current range volatility is elevated versus its medium baseline.
f=-p.pct_change(10).div(p.pct_change().rolling(20,min_periods=15).std(),axis=0).mul(short.div(long))
ics=[];ns=[];cov=[];turn=[];dates=[];sig=[]
for i in range(len(p)-10):
 if i<100 or p.index[i+10]>cut: continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  ics.append(z);ns.append(ok.sum());cov.append(ok.mean());dates.append(p.index[i]);sig.append(x)
  if len(sig)>1:
   q=sig[-2];o=x.notna()&q.notna();turn.append(float((x[o].rank(pct=True)-q[o].rank(pct=True)).abs().mean()))
a=np.array(ics);D=pd.DatetimeIndex(dates);print({'factor':'range_expansion_shock_reversal_10d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turn))})
for n,m in [('180',D>=pd.Timestamp('2030-01-01')),('360',D>=pd.Timestamp('2029-08-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m];print(n,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_2_20300822_range_expansion_shock_reversal_10d_ic.csv',index=False);pd.DataFrame(sig,index=dates,columns=U).to_csv('scripts/miner_2_20300822_range_expansion_shock_reversal_10d_signal.csv');np.savez('scripts/miner_2_20300822_range_expansion_shock_reversal_10d_artifact.npz',dates=np.array([str(x.date()) for x in dates]),signal=np.array(sig),assets=np.array(U))
