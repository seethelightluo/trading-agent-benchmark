import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-11-11')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill().bfill()
# VIX regime multiplier: short reversal is stronger in elevated volatility, modestly damped otherwise.
base=-p.pct_change(5)/(r.rolling(20,min_periods=20).std()*np.sqrt(5))
z=(vix-vix.rolling(252,min_periods=126).mean())/vix.rolling(252,min_periods=126).std()
f=(base.mul((1+0.5*z.clip(-2,2)), axis=0)).clip(-5,5).shift(1).rolling(3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
rows=[]
for i,d in enumerate(p.index[:-21]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 q=ic(f.loc[d],(p.shift(-10)/p-1).loc[d])
 if pd.notna(q):rows.append((d,q,int(f.loc[d].notna().sum())))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); zic=x.ic
print('dates',len(zic),'start',x.index.min().date() if len(x) else 'NA','end',x.index.max().date() if len(x) else 'NA','avg_n',x.n.mean() if len(x) else 0,'coverage',f.loc[x.index].notna().mean().mean() if len(x) else 0)
print('IC',zic.mean(),'ICIR',zic.mean()/zic.std(ddof=1),'hit',(zic>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:print('decay',h,np.nanmean([ic(f.loc[d],(p.shift(-h)/p-1).loc[d]) for d in x.index]))
for n,q in [('365',zic.tail(365)),('180',zic.tail(180)),('2032',zic[zic.index.to_series().dt.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_3_20321111_vix_regime_reversal_signal.csv');x.to_csv('scripts/miner_3_20321111_vix_regime_reversal_ic.csv')
