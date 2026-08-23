import pandas as pd,numpy as np,json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-09-01')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill().loc[:cut]; r=p.pct_change(); m=r.mean(axis=1)
ex=r.rolling(20,min_periods=20).sum().sub(m.rolling(20,min_periods=20).sum(),axis=0); rv=r.rolling(20,min_periods=20).std(); base=(ex/(rv*np.sqrt(20)+1e-12)).shift(1).rolling(3,min_periods=3).mean(); b=((r.rolling(20,min_periods=20).sum()>0).mean(axis=1).shift(1)-.5)*2
f=base.mul(b,axis=0)
def ic(a,b):
 ok=a.notna()&b.notna()
 return spearmanr(a[ok],b[ok]).statistic if ok.sum()>=8 and a[ok].nunique()>=3 and b[ok].nunique()>=3 else np.nan
fr=p.shift(-10)/p-1; rows=[]
for d in p.index:
 q=ic(f.loc[d],fr.loc[d])
 if d>=pd.Timestamp('2020-06-01') and pd.notna(q):rows.append((d,q,(f.loc[d].notna()&fr.loc[d].notna()).sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic;m=float(z.mean());ir=float(m/z.std(ddof=1));turn=float(f.rank(pct=True).diff().abs().mean().mean())
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',m,'ICIR',ir,'hit',(z>0).mean(),'turnover',turn)
for h in [1,5,20]:
 q=[ic(f.loc[d],(p.shift(-h)/p-1).loc[d]) for d in x.index];q=[v for v in q if pd.notna(v)];print('decay',h,float(np.mean(q)))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,float(q.mean()),float(q.mean()/q.std(ddof=1)),len(q))
f.loc[x.index].to_csv('scripts/miner_1_20320902_continuous_breadth_relative_momentum_signal.csv');x.to_csv('scripts/miner_1_20320902_continuous_breadth_relative_momentum_ic.csv')
print('METRICS',json.dumps({'ic':m,'icir':ir,'turnover':turn,'coverage':float(f.loc[x.index].notna().mean().mean()),'dates':len(z),'avg_n':float(x.n.mean()),'max_abs_library_correlation':None}))
