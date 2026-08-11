import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-10-07'
def make(h=1):
 out=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date')
  r=x.close.pct_change(); vol=r.rolling(20,min_periods=12).std()
  # recent shock reversal, scaled by each asset's trailing risk; winsorization is cross-sectional
  f=(-r/vol).clip(-4,4)
  y=x.close.shift(-h)/x.close-1
  out.append(pd.DataFrame({'date':x.index,'s':s,'f':f,'y':y}))
 return pd.concat(out,ignore_index=True).dropna()
def validate(a):
 out=[]; ns=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   c=spearmanr(g.f,g.y).statistic
   if pd.notna(c): out.append((d,c)); ns.append(len(g))
 z=pd.DataFrame(out,columns=['date','ic']).set_index('date'); q=z.ic
 ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank')
 turnover=ranks.diff().abs().mean(axis=1).mean()
 print('dates',len(q),'avg_n',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',turnover)
 print('coverage',len(a)/(sum(len(pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut')) for s in U)))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-10-07')]:
  v=z.loc[lo:hi].ic; print('regime',lo,hi,'n',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
 for h in [3,5,10]:
  zz=validate_h(make(h)); print('decay',h,'n',len(zz),'IC',zz.mean(),'ICIR',zz.mean()/zz.std(ddof=1))
 a[['date','s','f']].rename(columns={'s':'symbol','f':'signal'}).to_csv('scripts/miner_1_20261008_risk_scaled_short_reversal_signal.csv',index=False)
def validate_h(a):
 vals=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   c=spearmanr(g.f,g.y).statistic
   if pd.notna(c): vals.append(c)
 return pd.Series(vals)
validate(make(1))
