import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2032-08-12')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill();r=p.pct_change()
# In high cross-asset dispersion, fade each asset's risk-normalized 5d move.
ret5=r.rolling(5,min_periods=5).sum().shift(1);vol20=r.rolling(20,min_periods=20).std().shift(1)
disp=r.rolling(10,min_periods=10).std().mean(axis=1).shift(1);thr=disp.rolling(120,min_periods=60).quantile(.65).shift(1)
gate=(disp>thr).astype(float);f=(-ret5/(vol20*np.sqrt(5)+1e-12)).mul(gate,axis=0).rolling(2,min_periods=2).mean().shift(1)
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
rows=[]
for i,d in enumerate(p.index[:-20]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d])
 if pd.notna(q):rows.append((d,q,int((f.loc[d].notna()).sum())))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean())
print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=[ic(f.loc[d],(p.shift(-h)/p-1).loc[d]) for d in x.index];q=[a for a in q if pd.notna(a)];print('decay',h,np.mean(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_3_20320819_dispersion_reversal_signal.csv');x.to_csv('scripts/miner_3_20320819_dispersion_reversal_ic.csv')
