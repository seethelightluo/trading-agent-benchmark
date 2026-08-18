import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(f'{b}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); r=p.pct_change()
# Medium-horizon residual reversal: invert 60d return after removing each asset's own 20d volatility scale.
f=(-p.pct_change(60).div(r.rolling(20).std())).shift(1); fw=p.shift(-10).div(p)-1
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate medium reversal 60/vol20; dates',len(out),'assets',len(A),'avg_n',out.n.mean(),'coverage',out.n.mean()/15)
print('IC',out.ic.mean(),'ICIR',out.ic.mean()/out.ic.std(),'hit',(out.ic>0).mean())
for lo,hi in [('2025','2030-12-31'),('2031','2035-12-31'),('2035-06-01','2035-12-21')]:
 q=out.loc[lo:hi].ic; print('regime',lo,hi,'dates',len(q),'ic',q.mean(),'icir',q.mean()/q.std() if len(q)>1 else np.nan)
for h in [1,3,5,10,20]:
 fw=p.shift(-h).div(p)-1; rr=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'ic',np.nanmean(rr),'n',len(rr))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20351221_medium_reversal_signal.csv',index=False)
