import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-02-19')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change();
# lag-safe short reversal, volatility normalized and conditioned by cross-asset dispersion
vol=r.rolling(20,min_periods=20).std().shift(1); shock=r.rolling(3,min_periods=3).sum().shift(1)
disp=r.std(axis=1).rolling(20,min_periods=15).mean(); base=disp.rolling(120,min_periods=60).median(); stress=(disp/(base+1e-12)).clip(.5,3)
# modestly increase reversal in stressed regimes, preserving interpretability
f=-shock/(vol+1e-12)*(1+0.35*(stress-1).clip(-.5,1.0)).values[:,None]
f=pd.DataFrame(f,index=p.index,columns=U)
fr=p.shift(-10)/p-1; rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 a=f.loc[d]; b=fr.loc[d]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in x.index:
  a=f.loc[d];b=yy.loc[d];ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2028',z['2028']),('2029',z['2029']),('2030',z['2030']),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
f.loc[x.index].to_csv('scripts/miner_3_20320219_stress_reversal_signal.csv');x.to_csv('scripts/miner_3_20320219_stress_reversal_ic.csv')
