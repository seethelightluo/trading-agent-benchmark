import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-09-06'); O={}; C={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').sort_values('date').set_index('date'); O[s]=x.open; C[s]=x.close
op=pd.DataFrame(O); px=pd.DataFrame(C).sort_index(); prev=px.shift(1)
# Reversal of prior session open-to-close gap, lagged and observable at next decision.
gap=(op/prev-1); s=-gap.shift(1); fw=px.shift(-1)/px-1; rows=[]
for d in px.index:
 q=pd.DataFrame({'s':s.loc[d],'f':fw.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.s.nunique()>1 and q.f.nunique()>1: rows.append((d,spearmanr(q.s,q.f).statistic,len(q)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('factor gap_reversal_1d dates',len(z),'rows',int(z.n.sum()),'avgN',round(z.n.mean(),2),'coverage',round(s.notna().sum().sum()/s.size,4))
print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
for lab,m in [('2020-22',z.index<'2023-01-01'),('2023-25',(z.index>='2023-01-01')&(z.index<'2026-01-01')),('2026',(z.index>='2026-01-01')&(z.index<'2027-01-01')),('2027',(z.index>='2027-01-01')&(z.index<'2028-01-01')),('2028',z.index>='2028-01-01'),('recent180',z.index>=END-pd.Timedelta(days=180))]:
 q=z[m]; print(lab,'dates',len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>1 else 'insufficient')
rank=s.rank(axis=1,pct=True); t=[]
for i in range(1,len(rank)):
 a=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(a)>=8:t.append(1-a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
print('turnover_proxy %.6f observations %d'%(np.nanmean(t),len(t)))
out=s.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20280907_gap_reversal_signal.csv',index=False)
