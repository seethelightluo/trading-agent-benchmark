import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-11-13')
cl={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); cl[s]=d.close
p=pd.DataFrame(cl).sort_index(); r=p.pct_change()
# Volatility-adjusted trend consistency: signed 20d return weighted by fraction of positive daily moves.
ret=p.pct_change(20); pos=(r>0).rolling(20).mean(); f=ret*(0.5+pos)
dates=[];ics=[];ns=[];cov=[];turn=[];prev=None; sig=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  dates.append(p.index[i]);ics.append(z);ns.append(int(ok.sum()));cov.append(float(ok.mean()));sig.append(x.values)
  if prev is not None:
   oo=x.notna()&prev.notna(); turn.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(dates); a=np.array(ics)
def met(z): return (float(z.mean()),float(z.mean()/z.std(ddof=1))) if len(z)>1 else (np.nan,np.nan)
print({'factor':'consistency_weighted_momentum_20d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turn))})
for n,m in [('180',D>=pd.Timestamp('2030-05-01')),('360',D>=pd.Timestamp('2029-11-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01')),('recent60',D>=pd.Timestamp('2030-08-01'))]:
 z=a[m]; print(n,len(z),*met(z))
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_1_20301114_consistency_weighted_momentum_20d_ic.csv',index=False)
pd.DataFrame(sig,index=dates,columns=U).to_csv('scripts/miner_1_20301114_consistency_weighted_momentum_20d_signal.csv')
