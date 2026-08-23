import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-03-20'); H=5
C={}; Hh={}; Ll={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 C[s]=x.close; Hh[s]=x.high; Ll[s]=x.low
cl=pd.DataFrame(C).sort_index(); hi=pd.DataFrame(Hh).reindex(cl.index); lo=pd.DataFrame(Ll).reindex(cl.index); r=cl.pct_change()
# Range-expansion-conditioned shock reversal. All components lagged one session.
shock=-(cl/cl.shift(3)-1).shift(1)/(r.rolling(20).std().shift(1)*np.sqrt(3))
tr=(hi-lo)/cl
exp=(tr.rolling(5,min_periods=5).mean()/tr.rolling(40,min_periods=20).median()).shift(1).clip(.5,2.0)
sig=(shock*exp).clip(-8,8)
fr=cl.shift(-H)/cl-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(q): rows.append((dt,q,len(z)))
D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for lab,s in [('full',D),('recent180',D.tail(180)),('recent360',D.tail(360))]:
 a=s.ic; print(lab,'dates',len(a),'avg_n',round(s.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(sig.notna().sum(axis=1).mean()/15,4),'period',D.index.min().date(),D.index.max().date())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20290322_range_expansion_shock_signal.csv',index=False)
