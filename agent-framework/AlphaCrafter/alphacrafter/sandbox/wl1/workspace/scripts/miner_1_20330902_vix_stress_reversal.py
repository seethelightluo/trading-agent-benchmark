import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a) for a in assets],axis=1).sort_index().loc[:'2033-09-01']
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
raw=-(p.pct_change(10).sub(p.pct_change(10).mean(axis=1),axis=0))
stress=(vix>vix.rolling(120,min_periods=60).quantile(.65)).astype(float)
f=raw.mul(0.35+0.65*stress,axis=0).shift(1)
rows=[]
for i in range(len(p)-10):
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],spearmanr(x[ok],y[ok]).statistic,ok.sum(),stress.iloc[i]))
z=pd.DataFrame(rows,columns=['date','ic','n','stress']).set_index('date')
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.sum()/(len(z)*15))
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),f.diff().abs().mean().mean()))
for name,g in z.groupby((z.index.year//3)*3): print(name,len(g),g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1))
for h in [5,10,20]:
 vals=[]
 for i in range(len(p)-h):
  x=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic)
 print('horizon',h,'IC',np.nanmean(vals),'ICIR',np.nanmean(vals)/np.nanstd(vals,ddof=1),'dates',len(vals))
sig=f.copy(); sig.insert(0,'date',sig.index.strftime('%Y-%m-%d')); sig.to_csv('scripts/miner_1_20330902_vix_stress_reversal_signal.csv',index=False)
