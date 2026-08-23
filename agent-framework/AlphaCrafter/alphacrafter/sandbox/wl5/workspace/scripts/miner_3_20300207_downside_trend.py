import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']) for s in U}
cut=pd.Timestamp('2030-02-06')
D={s:d[d.date<=cut].copy() for s,d in D.items()}
# downside-risk-adjusted medium trend: 20d return divided by downside deviation of last 40d;
# cross-sectional demean makes the signal relative and causal.
rows=[]
for s,df in D.items():
 if df is None: continue
 x=df[['date','close','pct_change']].copy(); x['r20']=x.close.pct_change(20)
 neg=x.pct_change if False else x['pct_change'].where(x['pct_change']<0,0.0)
 x['dd40']=neg.rolling(40,min_periods=30).std()*np.sqrt(252)
 x['raw']=x.r20/x.dd40.replace(0,np.nan)
 x['symbol']=s; rows.append(x[['date','symbol','raw']])
z=pd.concat(rows).dropna(); p=z.pivot(index='date',columns='symbol',values='raw')
cl=pd.concat({s:D[s].set_index('date').close for s in D if D[s] is not None},axis=1).reindex(p.index)
for h in [5,10,20]:
  fr=cl.shift(-h)/cl-1; ics=[]
  for dt in p.index:
   a=p.loc[dt]; b=fr.loc[dt]; q=pd.concat([a,b],axis=1).dropna()
   if len(q)>=8: ics.append(q.iloc[:,0].corr(q.iloc[:,1]))
  a=np.asarray(ics); print(h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('rows',len(z),'instruments',p.shape[1],'coverage',z.groupby('date').size().mean()/15)
# turnover in rank signal
r=p.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
# save artifact for admitted horizon if any
out=p.copy(); out.index.name='date'; out.to_csv('scripts/miner_3_20300207_downside_trend_signal.csv')
print('period',z.date.min(),z.date.max())
