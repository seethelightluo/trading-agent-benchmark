import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U};p=pd.DataFrame(p).sort_index();r=p.pct_change()
# inverse downside semideviation (zeros for up days), plus modest 60d trend confirmation
neg=r.clip(upper=0); dd=np.sqrt((neg**2).rolling(30).mean()); trend=p.pct_change(60); f=(-dd)+.25*trend/r.rolling(30).std().replace(0,np.nan)
rows=[]
for t in f.index:
 x=f.loc[t];y=r.shift(-1).loc[t];ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append([t,ok.sum(),spearmanr(x[ok],y[ok]).statistic])
r2=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');z=r2.ic.dropna();print('dates',len(z),'avgN',r2.n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean());print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-24')]:
 q=r2.loc[a:b,'ic'].dropna();print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270325_downside_quality_signal.csv',index=False)
