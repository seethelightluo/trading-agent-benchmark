import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-11-24')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change(); neg=r.clip(upper=0); d=np.sqrt((neg**2).rolling(40,min_periods=30).mean())
# lower downside-risk rank, lagged and smoothed
f=(-d.rank(axis=1,pct=True)).shift(1).rolling(3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 return spearmanr(a[ok],b[ok]).statistic if ok.sum()>=8 and a[ok].nunique()>=3 and b[ok].nunique()>=3 else np.nan
rows=[]
for i,x in enumerate(p.index[:-21]):
 if x<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 q=ic(f.loc[x],(p.shift(-10)/p-1).loc[x])
 if pd.notna(q): rows.append((x,q,int(f.loc[x].notna().sum())))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=a.ic
print('dates',len(z),'avg_n',a.n.mean(),'coverage',f.loc[a.index].notna().mean().mean()); print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]: print('decay',h,np.nanmean([ic(f.loc[x],(p.shift(-h)/p-1).loc[x]) for x in a.index]))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[a.index].to_csv('scripts/miner_2_20321125_downsidevol_signal.csv');a.to_csv('scripts/miner_2_20321125_downsidevol_ic.csv')
