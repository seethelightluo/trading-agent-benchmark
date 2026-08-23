import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-07-25')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); vol=r.rolling(20).std(); ret15=px/px.shift(15)-1
breadth=(ret15>0).mean(axis=1); stress=(breadth<0.50).astype(float)
sig=-(ret15/vol).replace([np.inf,-np.inf],np.nan).mul(0.5+stress,axis=0).shift(1)
fw=px.shift(-1)/px-1; rows=[]
for d in px.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fw.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('candidate=stress50-conditioned volatility-scaled 15d reversal'); print('dates',len(z),'rows',int(z.n.sum()),'avgN',round(z.n.mean(),2),'coverage',round(sig.notna().sum().sum()/sig.size,4)); print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()))
for lab,m in [('2026',z.index>='2026-01-01'),('2027',(z.index>='2027-01-01')&(z.index<'2028-01-01')),('2028',z.index>='2028-01-01'),('recent180',z.index>=END-pd.Timedelta(days=180))]:
 q=z[m]; print(lab,'dates',len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>1 else 'insufficient')
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20280727_stress50_volscaled_reversal15_signal.csv',index=False)
