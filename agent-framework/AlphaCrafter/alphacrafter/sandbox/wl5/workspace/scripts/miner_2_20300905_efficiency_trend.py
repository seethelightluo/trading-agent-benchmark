import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
close=pd.DataFrame(C).sort_index(); r=close.pct_change(); ret20=close/close.shift(20)-1; path=r.abs().rolling(20).sum(); f=(ret20/path.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan); fr=close.shift(-10)/close-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.sum()/(len(x)*15));print('ic',x.ic.mean(),'icir',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-08-21')]:
 q=x.loc[a:b];print(a,'dates',len(q),'ic',q.ic.mean(),'icir',q.ic.mean()/q.ic.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.reset_index().to_csv('scripts/miner_2_20300905_efficiency_trend_signal.csv',index=False)
