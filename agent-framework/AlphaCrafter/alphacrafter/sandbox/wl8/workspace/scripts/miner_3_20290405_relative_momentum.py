import numpy as np,pandas as pd
END=pd.Timestamp('2029-04-05'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=10
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:END] for s in U}
cl=pd.DataFrame(P).sort_index(); r=cl.pct_change();
# Relative momentum: asset 20d return minus contemporaneous cross-asset median, lagged one day
raw=(cl/cl.shift(20)-1).shift(1); sig=raw.sub(raw.median(axis=1),axis=0)
fr=cl.shift(-H)/cl-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q): rows.append((dt,q,len(z)))
D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for lab,s in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360)),('2026',D.loc['2026']),('2027-28',D.loc['2027':'2028']),('2029',D.loc['2029'])]:
 a=s.ic
 if len(a): print(lab,'dates',len(a),'avg_n',round(s.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4),'period',D.index.min().date(),D.index.max().date())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20290405_relative_momentum_signal.csv',index=False)
