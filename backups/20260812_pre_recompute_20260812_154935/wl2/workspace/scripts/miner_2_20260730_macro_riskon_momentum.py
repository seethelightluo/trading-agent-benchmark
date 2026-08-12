import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.concat(D,axis=1).sort_index().loc[:'2026-07-15']; r=P.pct_change(20)
# risk-on regime: median equity momentum minus average yield-series momentum; sign lagged
risk=(r[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].median(axis=1)-r[['US10Y','CN10Y']].mean(axis=1)).shift(1)
sig=r.mul(np.sign(risk),axis=0).shift(1); fwd=P.pct_change().shift(-1)
def run(ret):
 z=[]; ns=[]
 for dt in sig.index:
  a=sig.loc[dt].dropna(); b=ret.loc[dt].dropna(); ix=a.index.intersection(b.index); a=a[ix]; b=b[ix]
  if len(ix)>=8 and a.nunique()>1 and b.nunique()>1:z.append((dt,spearmanr(a,b).statistic));ns.append(len(ix))
 q=pd.DataFrame(z,columns=['date','ic']).set_index('date');return q,ns
q,n=run(fwd)
for l,z in [('all',q),('20-22',q.loc['2020':'2022']),('23-24',q.loc['2023':'2024']),('25-26',q.loc['2025':'2026'])]:print(l,len(z),f'IC {z.ic.mean():.6f} ICIR {z.ic.mean()/z.ic.std(ddof=1):.6f} hit {(z.ic>0).mean():.3f}')
print('avg_n',np.mean(n),'coverage',np.mean(n)/15,'turnover',sig.rank(pct=True).diff().abs().mean(axis=1).loc[q.index].mean()/2)
for h in [5,10]:
 z,n=run(P.pct_change(h).shift(-h));print('h',h,len(z),f'IC {z.ic.mean():.6f} ICIR {z.ic.mean()/z.ic.std(ddof=1):.6f}')
