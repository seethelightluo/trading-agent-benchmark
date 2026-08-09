import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-12-16')
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); d=d.sort_values('date'); d=d[d.date<=END]
 rg=(d.high-d.low).replace(0,np.nan)
 f=(-(2*(d.close-d.low)/rg-1)).rolling(3,min_periods=3).mean()
 for dt,x in zip(d.date,f): rows.append((dt,s,x))
out=pd.DataFrame(rows,columns=['date','symbol','factor']).dropna()
out.to_csv('scripts/miner_1_20261217_clv3_signal.csv',index=False)
print('saved',len(out),'rows',out.date.min(),out.date.max())
# same-date next-observation return IC
ics=[]
for s,g in out.groupby('symbol'):
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv'); p.date=pd.to_datetime(p.date); p=p.sort_values('date'); p=p[p.date<=END]
 p['ret1']=p.close.pct_change().shift(-1)
 out.loc[out.symbol.eq(s),'ret1']=out.loc[out.symbol.eq(s),'date'].map(p.set_index('date').ret1)
for dt,g in out.groupby('date'):
 g=g.dropna(subset=['ret1'])
 if len(g)>=8: ics.append(g.factor.rank().corr(g.ret1.rank()))
a=np.array(ics); print('dates',len(a),'avgN',out.groupby('date').factor.count().mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
print('coverage',len(out)/(len(U)*len(set(out.date))))
print('turnover',out.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean().mean())
