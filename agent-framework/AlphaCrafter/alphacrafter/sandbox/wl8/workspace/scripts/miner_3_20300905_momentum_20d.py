import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-08-21'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'];px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); f=p.pct_change(20); dates=[];a=[];ns=[];cov=[];turn=[];ss=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  dates.append(p.index[i]);a.append(z);ns.append(ok.sum());cov.append(ok.mean());ss.append(x)
  if len(ss)>1:
   q=ss[-2];oo=x.notna()&q.notna();turn.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(a);D=pd.DatetimeIndex(dates);print({'factor':'momentum_20d_10d_research','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turn))})
for n,m in [('180',D>=pd.Timestamp('2030-02-21')),('360',D>=pd.Timestamp('2029-08-21')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01')),('recent',D>=pd.Timestamp('2030-04-01'))]:
 z=a[m];print(n,len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)))
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_3_20300905_momentum_20d_ic.csv',index=False);pd.DataFrame(ss,index=dates,columns=U).to_csv('scripts/miner_3_20300905_momentum_20d_signal.csv')
