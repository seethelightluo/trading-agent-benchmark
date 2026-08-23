import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-09-01')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
p=p.loc[:cut]; r=p.pct_change()
# Breadth-conditioned relative momentum: use lagged 20d relative strength; reverse its direction only when broad market is weak.
mkt=r.mean(axis=1)
excess=r.rolling(20,min_periods=20).sum().sub(mkt.rolling(20,min_periods=20).sum(),axis=0)
rv=r.rolling(20,min_periods=20).std()
base=(excess/(rv*np.sqrt(20)+1e-12)).shift(1).rolling(3,min_periods=3).mean()
breadth=(r.rolling(20,min_periods=20).sum()>0).mean(axis=1).shift(1)
f=base.mul(np.where(breadth>=0.5,1.0,-1.0),axis=0)
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
fr={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
rows=[]
for d in p.index:
 if d<pd.Timestamp('2020-06-01') or d not in f.index: continue
 q=ic(f.loc[d],fr[10].loc[d])
 if pd.notna(q): rows.append((d,q,(f.loc[d].notna()&fr[10].loc[d].notna()).sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
m=float(z.mean()); ir=float(m/z.std(ddof=1)); turn=float(f.rank(pct=True).diff().abs().mean().mean())
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',m,'ICIR',ir,'hit',(z>0).mean(),'turnover',turn)
for h in [1,5,10,20]:
 q=[ic(f.loc[d],fr[h].loc[d]) for d in x.index];q=[v for v in q if pd.notna(v)];print('decay',h,float(np.mean(q)))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]: print(n,float(q.mean()),float(q.mean()/q.std(ddof=1)),len(q))
f.loc[x.index].to_csv('scripts/miner_1_20320902_breadth_conditioned_relative_momentum_signal.csv'); x.to_csv('scripts/miner_1_20320902_breadth_conditioned_relative_momentum_ic.csv')
print('METRICS',json.dumps({'ic':m,'icir':ir,'turnover':turn,'coverage':float(f.loc[x.index].notna().mean().mean()),'dates':len(z),'avg_n':float(x.n.mean()),'max_abs_library_correlation':None}))
