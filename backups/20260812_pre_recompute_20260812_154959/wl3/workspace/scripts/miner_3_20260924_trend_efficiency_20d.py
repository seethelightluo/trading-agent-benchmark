import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-09-23'; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date'); D[s]=x.close
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Trend-efficiency: directional 20d move relative to realized path, bounded [-1,1].
f=r.rolling(20,min_periods=15).sum()/(r.abs().rolling(20,min_periods=15).sum()+1e-12)
def calc(yy,ff=f):
 vals=[]; ns=[]
 for d in ff.index.intersection(yy.index):
  a=pd.DataFrame({'f':ff.loc[d],'y':yy.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1: vals.append(spearmanr(a.f,a.y).statistic); ns.append(len(a))
 return np.array(vals),ns
y=p.shift(-1)/p-1
q,ns=calc(y); print('candidate trend_efficiency_20d cutoff',cut,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
for name,lo,hi in [('early','2020-06','2022-12'),('mid','2023-01','2024-12'),('late','2025-01','2026-09')]:
 qq,_=calc(y.loc[lo:hi],f.loc[lo:hi]); print('regime',name,len(qq),round(qq.mean(),6),round(qq.mean()/qq.std(ddof=1),6))
for h in [3,5,10]:
 vv,_=calc(p.shift(-h)/p-1); print('decay',h,len(vv),round(vv.mean(),6),round(vv.mean()/vv.std(ddof=1),6))
f.rename_axis('date').to_csv('scripts/miner_3_20260924_trend_efficiency_20d_signal.csv')
print('period',p.index.min(),p.index.max())
