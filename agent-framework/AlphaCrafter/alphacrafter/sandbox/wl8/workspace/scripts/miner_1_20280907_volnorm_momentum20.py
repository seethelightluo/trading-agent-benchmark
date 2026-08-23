import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-09-06')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1)
# Intermediate trend: trailing 20d return, volatility-normalized, lagged.
s=r.rolling(20,min_periods=20).sum().shift(1)/vol
fw=px.shift(-1)/px-1; rows=[]
for d in px.index:
 g=pd.DataFrame({'s':s.loc[d],'f':fw.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('factor volnorm_momentum_20d dates',len(z),'rows',int(z.n.sum()),'avgN',round(z.n.mean(),2),'coverage',round(s.notna().sum().sum()/s.size,4))
print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
for lab,m in [('2020-22',z.index<'2023-01-01'),('2023-25',(z.index>='2023-01-01')&(z.index<'2026-01-01')),('2026',(z.index>='2026-01-01')&(z.index<'2027-01-01')),('2027',(z.index>='2027-01-01')&(z.index<'2028-01-01')),('2028',z.index>='2028-01-01'),('recent180',z.index>=END-pd.Timedelta(days=180))]:
 q=z[m]; print(lab,'dates',len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>1 else 'insufficient')
rank=s.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rank)):
 a=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(a)>=8: turn.append(1-a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
print('turnover_proxy %.6f observations %d'%(np.nanmean(turn),len(turn)))
out=s.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20280907_volnorm_momentum20_signal.csv',index=False)
