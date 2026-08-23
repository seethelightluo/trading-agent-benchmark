import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2034-03-01'); px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end]
 px[s]=d.close
P=pd.DataFrame(px); ret=P.pct_change(); market=ret.mean(axis=1)
# Residualized 20d momentum: cumulative asset return less contemporaneous equal-weight market return, volatility scaled.
asset20=P/P.shift(20)-1; mkt20=P.mean(axis=1)/P.mean(axis=1).shift(20)-1
# use mean of individual 20d returns to avoid price-level mixing
mkt20=asset20.mean(axis=1)
vol=ret.rolling(30).std()*np.sqrt(252)
F=(asset20.sub(mkt20,axis=0))/vol

def run(h):
 rows=[]
 for s in U:
  z=pd.DataFrame({'f':F[s],'p':P[s]}); z['r']=z.p.shift(-h)/z.p-1
  z=z.dropna()
  rows += [(dt,s,x.f,x.r) for dt,x in z.iterrows()]
 a=pd.DataFrame(rows,columns=['date','s','f','r']); out=[]; ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   out.append(spearmanr(g.f,g.r).statistic); ns.append(len(g))
 o=np.array(out); return len(o),np.mean(ns),np.mean(o),np.mean(o)/np.std(o,ddof=1)*np.sqrt(252),np.mean(o>0),a
for h in [5,10,20,40]: print('H',h,run(h)[:5])
n,an,ic,ir,hit,a=run(10)
ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank')
print('DETAIL dates',n,'avg_n',an,'coverage',len(a)/(n*15),'turnover',ranks.diff().abs().mean().mean(),'period',a.date.min().date(),a.date.max().date())
print('annual',a.assign(ic=a.groupby('date').f.transform(lambda x: np.nan)).head(0))
