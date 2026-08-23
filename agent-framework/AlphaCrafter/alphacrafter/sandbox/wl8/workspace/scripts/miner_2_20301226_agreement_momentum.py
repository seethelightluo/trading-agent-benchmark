import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-12-25')
cl={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(cl).sort_index(); r=p.pct_change()
# Medium-horizon momentum, normalized by realized risk and damped when recent reversal conflicts.
vol=r.rolling(30).std(); mom60=p.pct_change(60); mom20=p.pct_change(20)
f=(0.65*mom60+0.35*mom20)/vol
# require agreement: reduce signals whose 20d and 60d momentum have opposite signs
f=f.where(np.sign(mom60)==np.sign(mom20),0.25*f)
dates=[];ics=[];ns=[];cov=[];turn=[];prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  dates.append(p.index[i]);ics.append(z);ns.append(int(ok.sum()));cov.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna(); turn.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(dates); a=np.array(ics)
print({'dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':np.mean(ns),'coverage':np.mean(cov),'ic':np.mean(a),'icir':np.mean(a)/np.std(a,ddof=1),'hit':np.mean(a>0),'turnover':np.mean(turn)})
for n,m in [('180',D>=pd.Timestamp('2030-06-01')),('360',D>=pd.Timestamp('2029-12-01')),('2030',D>=pd.Timestamp('2030-01-01')),('recent60',D>=pd.Timestamp('2030-09-15'))]:
 z=a[m]; print(n,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_2_20301226_agreement_momentum_ic.csv',index=False)
pd.DataFrame([f.loc[d].values for d in dates],index=dates,columns=U).to_csv('scripts/miner_2_20301226_agreement_momentum_signal.csv')
