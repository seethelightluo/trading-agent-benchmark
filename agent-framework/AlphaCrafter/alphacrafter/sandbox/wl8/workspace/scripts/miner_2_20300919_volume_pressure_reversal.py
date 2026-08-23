import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-09-05'); closes={}; vols={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); closes[s]=d.close[d.index<=cut]; vols[s]=d.volume[d.index<=cut]
p=pd.DataFrame(closes).sort_index(); v=pd.DataFrame(vols).reindex(p.index); ret=p.pct_change();
# contrarian medium-term return, strengthened by unusually high recent activity
f=-ret.rolling(20,min_periods=15).sum()*(v.rolling(5,min_periods=3).mean()/v.rolling(60,min_periods=30).mean()).clip(0.25,4)
ics=[];ns=[];cov=[];turn=[];dates=[];sig=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  ics.append(z);ns.append(ok.sum());cov.append(ok.mean());dates.append(p.index[i]);sig.append(x)
  if len(sig)>1:
   q=sig[-2];oo=x.notna()&q.notna();turn.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(ics);D=pd.DatetimeIndex(dates)
print({'factor':'volume_pressure_reversal_20d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turn))})
for name,m in [('180',D>=pd.Timestamp('2030-03-01')),('360',D>=pd.Timestamp('2029-09-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01')),('recent',D>=pd.Timestamp('2030-05-01'))]:
 z=a[m];print(name,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_2_20300919_volume_pressure_reversal_ic.csv',index=False)
pd.DataFrame(sig,index=dates,columns=U).to_csv('scripts/miner_2_20300919_volume_pressure_reversal_signal.csv')
