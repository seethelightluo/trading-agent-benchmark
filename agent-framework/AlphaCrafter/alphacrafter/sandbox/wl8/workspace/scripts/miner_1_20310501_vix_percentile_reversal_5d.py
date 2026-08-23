import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
r=p.pct_change(); rv=r.rolling(20,min_periods=15).std()
raw=-p.pct_change(5)/(rv*np.sqrt(5))
# Continuous, lagged cross-time VIX percentile; retain baseline signal in calm regimes.
q=v.rolling(120,min_periods=60).rank(pct=True).shift(1).clip(.05,.95)
f=raw.mul((0.75+0.75*q), axis=0)
rows=[]; decay={h:[] for h in [1,5,10,20]}; turns=[]
for i in range(len(p)-20):
 d=p.index[i]
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>pd.Timestamp('2031-04-30'): continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z):
   rows.append((d,ok.sum(),z))
   for h in decay:
    yy=p.iloc[i+h]/p.iloc[i]-1; oo=x.notna()&yy.notna()
    if oo.sum()>=8: decay[h].append(spearmanr(x[oo],yy[oo]).statistic)
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean(),f.rank(axis=1).diff().abs().stack().mean()/14))
for lab,x in [('180',a.tail(180)),('360',a.tail(360)),('2030',a[pd.to_datetime(a.index).year==2030]),('2031',a[pd.to_datetime(a.index).year==2031]),('60',a.tail(60))]: print(lab,len(x),'IC %.6f ICIR %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1)))
for h,z in decay.items(): print('decay',h,len(z),'IC %.6f ICIR %.6f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1)))
f.to_csv('scripts/miner_1_20310501_vix_percentile_reversal_5d_signal.csv'); a.to_csv('scripts/miner_1_20310501_vix_percentile_reversal_5d_ic.csv')
