import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill()
r=p.pct_change(); rv=r.rolling(20,min_periods=10).std(); raw=-p.pct_change(5)/(rv*np.sqrt(5)); med=v.rolling(120,min_periods=60).median()
stress=(v>med).astype(float); f=raw*(1+0.75*stress)
rows=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>pd.Timestamp('2031-08-21'): continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): rows.append((p.index[i],ok.sum(),z))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); a.index=pd.to_datetime(a.index)
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean(),f.rank(axis=1).diff().abs().stack().mean()/14))
for lab,x in [('180',a.tail(180)),('360',a.tail(360)),('2030',a[a.index.year==2030]),('2031',a[a.index.year==2031]),('60',a.tail(60))]: print(lab,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1))
for h in [1,5,10,20]:
 q=[]
 for i in range(len(p)-h):
  if p.index[i] not in a.index: continue
  x=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1;ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
f.to_csv('scripts/miner_1_20310821_vix_conditioned_reversal_signal.csv'); a.to_csv('scripts/miner_1_20310821_vix_conditioned_reversal_ic.csv')
