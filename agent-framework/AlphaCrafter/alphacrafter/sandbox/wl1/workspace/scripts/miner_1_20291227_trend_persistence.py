import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2029-12-27')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[s]=d[d.index<=cutoff]
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); mom=P/P.shift(20)-1; persist=(r.gt(0).rolling(20,min_periods=18).mean()-.5)*2; vol=r.rolling(40,min_periods=35).std()*np.sqrt(252); f=(mom*persist/(vol+1e-8)).shift(1); fr=P.shift(-20)/P-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('universe',len(U),'dates',len(q),'range',q.index.min(),q.index.max(),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*len(U)))
for label,sub in [('all',q),('2020_25',q[q.index<'2026-01-01']),('2026+',q[q.index>='2026-01-01']),('2028+',q[q.index>='2028-01-01']),('2029YTD',q[q.index>='2029-01-01'])]:
 ic=sub.ic.mean(); icir=ic/(sub.ic.std(ddof=1)/np.sqrt(len(sub))) if len(sub)>1 else np.nan
 print(label,'dates',len(sub),'IC %.6f ICIR %.6f hit %.4f'%(ic,icir,(sub.ic>0).mean()))
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
for h in [5,10,20,40]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),len(rr))
q.to_csv('scripts/miner_1_20291227_trend_persistence_signal.csv')
