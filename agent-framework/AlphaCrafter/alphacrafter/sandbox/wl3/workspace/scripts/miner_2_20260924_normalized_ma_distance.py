import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-09-23')
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'];return d[d.index<=cut]
p=pd.DataFrame({s:load(s) for s in U}).sort_index().ffill(); r=p.pct_change();
# distance below/above medium moving average, with risk normalization; negative distance is reversal score
f=-(p/p.rolling(20,min_periods=15).mean()-1)/(r.rolling(20,min_periods=15).std()+1e-8); y=r.shift(-1)
def calc(ff,yy):
 q=[];n=[]
 for d in ff.index:
  a=pd.DataFrame({'f':ff.loc[d],'y':yy.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1:q.append(spearmanr(a.f,a.y).statistic);n.append(len(a))
 q=np.asarray(q);return len(q),np.mean(n),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('candidate=normalized_ma_distance','assets',len(U),'rows',len(p))
for h in [1,3,5,10]:print('decay',h,calc(f,p.pct_change(h).shift(-h)))
for nm,lo,hi in [('early','2020-06','2022-12'),('mid','2023-01','2024-12'),('late','2025-01','2026-09-23')]:print('regime',nm,calc(f.loc[lo:hi],y.loc[lo:hi]))
print('coverage',f.notna().sum().sum()/f.size,'rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.rename_axis('date').to_csv('scripts/miner_2_20260924_normalized_ma_distance_signal.csv')
