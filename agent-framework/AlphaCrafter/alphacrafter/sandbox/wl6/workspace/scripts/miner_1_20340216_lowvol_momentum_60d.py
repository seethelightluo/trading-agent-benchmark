import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2034-02-15'); xs={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end]
 r=d.close.pct_change(); f=(d.close/d.close.shift(60)-1)/(r.rolling(60).std()*np.sqrt(252)); xs[s]=pd.DataFrame({'f':f,'p':d.close})
def run(h):
 rows=[]
 for s,z in xs.items():
  z=z.dropna().copy(); z['r']=z.p.shift(-h)/z.p-1
  for dt,x in z.dropna().iterrows(): rows.append((dt,s,x.f,x.r))
 a=pd.DataFrame(rows,columns=['date','s','f','r']); out=[]; ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: out.append(spearmanr(g.f,g.r).statistic);ns.append(len(g))
 o=np.array(out); return len(o),np.mean(ns),np.mean(o),np.mean(o)/np.std(o,ddof=1)*np.sqrt(252),np.mean(o>0),a
for h in [5,10,20,40]: print(h,run(h)[:5])
n,an,ic,ir,hit,a=run(10); ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank'); print('turnover',ranks.diff().abs().mean().mean(),'coverage',len(a)/(n*15),'period',a.date.min(),a.date.max())
