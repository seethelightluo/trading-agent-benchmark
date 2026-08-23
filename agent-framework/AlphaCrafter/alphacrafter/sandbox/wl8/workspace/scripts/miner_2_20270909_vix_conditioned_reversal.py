import numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); P[a]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.concat(P,axis=1).sort_index();
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float).reindex(p.index).ffill()
r3=p.shift(1).pct_change(3); vc=v.shift(1).pct_change(5); med=vc.rolling(252,min_periods=60).median(); sd=vc.rolling(252,min_periods=60).std(); shock=((vc-med)/(sd+1e-12)).clip(-3,3)
f=r3.mul(-(1+0.75*shock.clip(lower=0)),axis=0); fr=p.pct_change().shift(-1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def met(q): return len(q),q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12),(q.ic>0).mean()
print('dates',len(r),'rows',r.n.sum(),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15)); print('all',met(r))
for label,mask in [('2020-22',r.index<'2023'),('2023-25',(r.index>='2023')&(r.index<'2026')),('2026',r.index.year==2026),('2027',r.index.year==2027),('recent90',r.index>=pd.Timestamp('2027-06-01'))]: print(label,met(r[mask]))
out=f.stack().rename('signal').reset_index().rename(columns={'level_1':'symbol'}); out.to_csv('scripts/miner_2_20270909_vix_conditioned_reversal_signal.csv',index=False); print('artifact_rows',len(out))
