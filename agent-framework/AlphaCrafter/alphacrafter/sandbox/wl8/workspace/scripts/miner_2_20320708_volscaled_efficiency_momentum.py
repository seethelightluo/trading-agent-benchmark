import pandas as pd,numpy as np,json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-07-08')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
r=p.pct_change(); lagr=r.shift(1)
mom=p.pct_change(20).shift(1); path=lagr.abs().rolling(20,min_periods=20).sum(); eff=(mom/(path+1e-12)).clip(-1,1)
vol=lagr.rolling(20,min_periods=20).std(); f=(mom*eff/(vol+1e-8)).rolling(3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
rows=[]; fr={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
for i,d in enumerate(p.index):
 if d<pd.Timestamp('2020-06-01') or d>cut or i+10>=len(p):continue
 q=ic(f.loc[d],fr[10].loc[d]);
 if pd.notna(q):rows.append((d,q,(f.loc[d].notna()&fr[10].loc[d].notna()).sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=[ic(f.loc[d],fr[h].loc[d]) for d in x.index];q=[v for v in q if pd.notna(v)];print('decay',h,np.mean(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_2_20320708_volscaled_efficiency_signal.csv');x.to_csv('scripts/miner_2_20320708_volscaled_efficiency_ic.csv')
print('coverage',float(f.loc[x.index].notna().mean().mean()))
