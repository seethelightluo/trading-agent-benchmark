import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-11-27')
cl={};vo={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date');cl[s]=d.close;vo[s]=d.volume
p=pd.DataFrame(cl).sort_index();v=pd.DataFrame(vo).reindex(p.index);r=p.pct_change(5);shock=np.log(v.rolling(3).mean()/v.rolling(30).mean()).clip(-3,3);f=-r*(1+0.35*shock)
dates=[];ics=[];ns=[];cov=[];turn=[];prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  dates.append(p.index[i]);ics.append(z);ns.append(int(ok.sum()));cov.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna();turn.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(dates);a=np.array(ics)
print({'dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':np.mean(ns),'coverage':np.mean(cov),'ic':np.mean(a),'icir':np.mean(a)/np.std(a,ddof=1),'hit':np.mean(a>0),'turnover':np.mean(turn)})
for n,m in [('180',D>=pd.Timestamp('2030-05-01')),('360',D>=pd.Timestamp('2029-11-01')),('2030',D>=pd.Timestamp('2030-01-01')),('recent60',D>=pd.Timestamp('2030-08-15'))]:
 z=a[m];print(n,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1))
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_2_20301128_volume_shock_reversal_ic.csv',index=False)
pd.DataFrame([f.loc[d].values for d in dates],index=dates,columns=U).to_csv('scripts/miner_2_20301128_volume_shock_reversal_signal.csv')
