import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-12-17'); rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=end].set_index('date'); r=d.close.pct_change()
 rows.append(pd.DataFrame({'r':r,'symbol':s}))
x=pd.concat(rows).reset_index(); wide=x.pivot(index='date',columns='symbol',values='r').sort_index()
# Equal-weight benchmark is formed from same-day returns; beta is estimated only from prior 60 completed observations.
bench=wide.mean(axis=1); factors=[]
for s in U:
 rr=wide[s]; cov=rr.rolling(60,min_periods=30).cov(bench).shift(1); var=bench.rolling(60,min_periods=30).var().shift(1)
 beta=cov/(var+1e-12); resid=rr-beta*bench
 rv=resid.rolling(20,min_periods=10).std().shift(1)
 # 3-session residual reversal, with all inputs lagged at decision date
 f=-(resid.shift(1)+resid.shift(2)+resid.shift(3))/(rv*np.sqrt(3)+1e-12)
 y=rr.shift(-1)
 factors.append(pd.DataFrame({'date':wide.index,'factor':f.values,'y':y.values,'symbol':s}))
z=pd.concat(factors,ignore_index=True); out=[]
for dt,g in z.groupby('date'):
 g=g.dropna()
 if len(g)>=8 and g.factor.nunique()>1 and g.y.nunique()>1:
  out.append((dt,spearmanr(g.factor,g.y).statistic,len(g)))
a=pd.DataFrame(out,columns=['date','ic','n']); q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),2),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',z.dropna().shape[0]/z.shape[0])
print('regimes',a.assign(reg=pd.cut(a.date.dt.year,[2019,2022,2024,2026,2027])).groupby('reg',observed=True).ic.mean().to_dict())
r=z.dropna().pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('turnover',r.diff().abs().mean(axis=1).mean())
for h in [5,10]:
 vals=[]
 for s in U:
  rr=wide[s]; vals.append(pd.DataFrame({'date':wide.index,'factor':factors[U.index(s)].factor.values,'y':rr.shift(-h).values/rr.shift(-1).values-1}))
 zz=pd.concat(vals,ignore_index=True); oo=[]
 for dt,g in zz.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.factor.nunique()>1 and g.y.nunique()>1: oo.append(spearmanr(g.factor,g.y).statistic)
 print('horizon',h,'dates',len(oo),'IC',np.mean(oo),'ICIR',np.mean(oo)/np.std(oo,ddof=1))
z.to_csv('scripts/miner_2_20261217_beta_neutral_residual_signal.csv',index=False)
print('period',z.date.min(),z.date.max(),'symbols',z.symbol.nunique())
