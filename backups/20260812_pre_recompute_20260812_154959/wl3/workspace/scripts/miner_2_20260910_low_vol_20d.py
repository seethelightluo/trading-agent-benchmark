import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-09-09')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();f=-r.rolling(20,min_periods=15).std();y=p.pct_change().shift(-1)
def go(yy,ff=f):
 q=[];n=[]
 for d in ff.index:
  a=pd.DataFrame({'f':ff.loc[d],'y':yy.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1:q.append(spearmanr(a.f,a.y).statistic);n.append(len(a))
 return np.array(q),n
q,n=go(y);print('candidate=low_vol_20d','dates',len(q),'avg_n',np.mean(n),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,lo,hi in [('early','2020-06','2022-12'),('mid','2023-01','2024-12'),('late','2025-01','2026-09')]:
 z,_=go(y.loc[lo:hi],f.loc[lo:hi]);print('regime',name,len(z),z.mean(),z.mean()/z.std(ddof=1))
for h in [3,5,10]:
 z,_=go(p.pct_change(h).shift(-h));print('decay',h,len(z),z.mean(),z.mean()/z.std(ddof=1))
f.rename_axis('date').to_csv('scripts/miner_2_20260910_low_vol_20d_signal.csv')
