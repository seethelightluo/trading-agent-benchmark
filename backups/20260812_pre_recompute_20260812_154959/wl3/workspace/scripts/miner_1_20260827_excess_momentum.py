import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-08-26'
def make(h):
 z=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date'); r=x.close.pct_change(); f=r.rolling(60,min_periods=40).sum()-r.rolling(5,min_periods=5).sum(); f=f/(r.rolling(60,min_periods=40).std()*np.sqrt(60)+1e-12); y=x.close.shift(-h)/x.close-1; z.append(pd.DataFrame({'date':x.date,'symbol':s,'signal':f,'fwd':y}))
 return pd.concat(z,ignore_index=True).dropna()
def run(a):
 q=[]; ns=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1:
   c=spearmanr(g.signal,g.fwd).statistic
   if pd.notna(c):q.append(c);ns.append(len(g))
 q=np.array(q); return len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
for h in [1,3,5,10]: print(h,run(make(h)))
a=make(1); ranks=a.assign(rank=a.groupby('date').signal.rank(pct=True)).pivot(index='date',columns='symbol',values='rank'); print('turnover',ranks.diff().abs().mean(axis=1).mean(),'coverage',len(a)/(sum(len(pd.read_csv('../persistent/stock_data/'+s+'.csv')) for s in U)))
