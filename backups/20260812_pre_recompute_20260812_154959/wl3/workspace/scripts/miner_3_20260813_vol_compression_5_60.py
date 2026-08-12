import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-08-12'; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date');D[s]=x.close
p=pd.DataFrame(D).sort_index();r=p.pct_change(); f=-(r.rolling(5,min_periods=5).std()/(r.rolling(60,min_periods=30).std()+1e-12)); y=p.shift(-1)/p-1
def calc(yy):
 vals=[];ns=[]
 for d in f.index.intersection(yy.index):
  a=pd.DataFrame({'f':f.loc[d],'y':yy.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1: vals.append(spearmanr(a.f,a.y).statistic);ns.append(len(a))
 return np.array(vals),ns
q,ns=calc(y); print('candidate vol_compression_5_60 cutoff',cut,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for name,lo,hi in [('early','2020-06','2022-12'),('mid','2023-01','2024-12'),('late','2025-01','2026-08')]:
 yy2=y.loc[lo:hi]; ff=f.loc[lo:hi]; old=f; f=ff; qq,_=calc(yy2); f=old; print('regime',name,len(qq),qq.mean(),qq.mean()/qq.std(ddof=1))
for h in [3,5,10]:
 yy2=p.shift(-h)/p-1; vv,_=calc(yy2); print('decay',h,len(vv),vv.mean(),vv.mean()/vv.std(ddof=1))
print('period',p.index.min(),p.index.max())
f.rename_axis('date').to_csv('scripts/miner_3_20260813_vol_compression_5_60_signal.csv')
