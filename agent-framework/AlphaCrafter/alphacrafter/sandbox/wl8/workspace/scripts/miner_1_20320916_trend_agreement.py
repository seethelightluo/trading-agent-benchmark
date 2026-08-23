import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-09-15')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20,min_periods=20).std().shift(1)
# Agreement-weighted trend: lagged 60d risk-adjusted trend, multiplied by agreement of 20/60/120d return signs.
a=[]
for w in [20,60,120]: a.append(r.rolling(w,min_periods=w).sum().shift(1))
ag=(np.sign(a[0])+np.sign(a[1])+np.sign(a[2]))/3
f=(a[1]/vol)*ag

def ic(x,y):
 ok=x.notna()&y.notna()
 if ok.sum()<8 or x[ok].nunique()<3 or y[ok].nunique()<3:return np.nan
 return spearmanr(x[ok],y[ok]).statistic
rows=[]
for i,d in enumerate(p.index[:-20]):
 if d<pd.Timestamp('2020-07-01') or p.index[i+10]>cut: continue
 q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d])
 if pd.notna(q): rows.append((d,q,int(f.loc[d].notna().sum())))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean())
print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]: print('decay',h,np.nanmean([ic(f.loc[d],(p.shift(-h)/p-1).loc[d]) for d in x.index]))
for name,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]: print(name,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_1_20320916_trend_agreement_signal.csv');x.to_csv('scripts/miner_1_20320916_trend_agreement_ic.csv')
