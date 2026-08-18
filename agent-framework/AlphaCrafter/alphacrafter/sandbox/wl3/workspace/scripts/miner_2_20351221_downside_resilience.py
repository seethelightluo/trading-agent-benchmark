import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
prices=pd.DataFrame({a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}).sort_index()
r=prices.pct_change()
# Downside-volatility resilience: medium-horizon return penalized only by realized negative-return risk.
down=r.where(r<0).rolling(40,min_periods=15).std()
raw=prices.pct_change(20).div(down.replace(0,np.nan))
f=raw.shift(1)
rows=[]
for dt in f.index:
 y=prices.shift(-10).div(prices)-1
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate downside-volatility resilience 20/40; dates',len(out),'assets',len(assets),'avg_n',out.n.mean(),'coverage',out.n.mean()/15)
print('IC',out.ic.mean(),'ICIR',out.ic.mean()/out.ic.std(),'hit',(out.ic>0).mean())
for lo,hi in [('2020','2024-12-31'),('2025','2030-12-31'),('2031','2035-12-31'),('2035-06-01','2035-12-21')]:
 q=out.loc[lo:hi].ic; print('regime',lo,hi,'dates',len(q),'ic',q.mean(),'icir',q.mean()/q.std() if len(q)>1 else np.nan)
for h in [1,3,5,10,20]:
 fw=prices.shift(-h).div(prices)-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'ic',np.nanmean(rr),'n',len(rr))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
# recoverable signal artifact
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20351221_downside_resilience_signal.csv',index=False)
