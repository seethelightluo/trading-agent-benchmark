import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-08-26'
def make(h=1):
 out=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date')
  r=x.close.pct_change(); v=np.log(x.volume.replace(0,np.nan)); vz=(v-v.rolling(20,min_periods=10).mean())/(v.rolling(20,min_periods=10).std()+1e-12)
  # Contrarian one-day return, amplified only by unusually high turnover; bounded to avoid crypto volume outliers
  f=-r*(1+0.5*vz.clip(-1,2)); y=x.close.shift(-h)/x.close-1
  out.append(pd.DataFrame({'date':x.index,'s':s,'f':f,'y':y}))
 return pd.concat(out,ignore_index=True).dropna()
def ics(a):
 out=[]; ns=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:
   c=spearmanr(g.f,g.y).statistic
   if pd.notna(c):out.append((d,c));ns.append(len(g))
 return pd.DataFrame(out,columns=['date','ic']).set_index('date'),ns
a=make(); z,ns=ics(a); q=z.ic
ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank')
print('factor volume_amplified_reversal_1d cutoff',cut,'dates',len(q),'avg_n',np.mean(ns),'coverage',len(a)/(sum(len(pd.read_csv('../persistent/stock_data/'+s+'.csv')) for s in U)),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',ranks.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-08-26')]:
 v=z.loc[lo:hi].ic; print('regime',lo,hi,'n',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1) if len(v)>1 else np.nan)
for h in [3,5,10]:
 zz,nn=ics(make(h)); print('decay',h,'n',len(zz),'IC',zz.ic.mean(),'ICIR',zz.ic.mean()/zz.ic.std(ddof=1))
a[['date','s','f']].rename(columns={'s':'symbol','f':'signal'}).to_csv('scripts/miner_3_20260827_volume_amplified_reversal_1d_signal.csv',index=False)
