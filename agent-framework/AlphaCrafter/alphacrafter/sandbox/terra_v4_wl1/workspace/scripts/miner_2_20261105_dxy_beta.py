import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-11-05'); base='../persistent/stock_data'
macro=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]

def one_asset(s,w=60):
 p=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut].dropna()
 x=pd.concat([p.rename('p'),macro.rename('m')],axis=1,join='inner').dropna()
 r=x.p.pct_change(); mr=x.m.pct_change()
 beta=r.rolling(w,min_periods=max(20,int(w*.75))).cov(mr)/mr.rolling(w,min_periods=max(20,int(w*.75))).var()
 # next observation for this asset, strictly after signal date
 fwd=p.shift(-1)/p-1
 return (-beta).rename('factor').to_frame().join(fwd.rename('fwd')).dropna()

def validate(w):
 A={s:one_asset(s,w) for s in U}; rows=[]
 dates=sorted(set().union(*[set(a.index) for a in A.values()]))
 for dt in dates:
  vals=[(s,A[s].loc[dt,'factor'],A[s].loc[dt,'fwd']) for s in U if dt in A[s].index]
  if len(vals)>=8:
   z=pd.DataFrame(vals,columns=['s','f','r']).dropna()
   if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1: rows.append((dt,len(z),spearmanr(z.f,z.r).statistic))
 d=pd.DataFrame(rows,columns=['date','n','ic']); q=d.ic
 return d,q,A

d,q,A=validate(60)
print('cutoff',cut.date(),'dates',len(q),'avgN',d.n.mean(),'coverage',d.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for yr,g in d.groupby(d.date.dt.year): print('year',yr,'obs',len(g),'IC',g.ic.mean(),'ICIR',g.ic.mean()/g.ic.std(ddof=1))
# rank turnover using consecutive valid dates, within each asset signal ranks cross-section
wide=pd.DataFrame({s:a.factor for s,a in A.items()}); ranks=wide.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
for w in [30,90,120]:
 dd,qq,_=validate(w); print('window',w,'IC',qq.mean(),'ICIR',qq.mean()/qq.std(ddof=1),'obs',len(qq))
print('n instruments',len(U),'asset coverage',sum(len(a) for a in A.values())/sum(len(pd.read_csv(f'{base}/{s}.csv')) for s in U))
