import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date <= @cut').set_index('date').sort_index() for s in U}
rows=[]
for s,x in D.items():
 ret=x.close.pct_change(); body=-(x.close-x.open)/x.open; rng=(x.high-x.low).replace(0,np.nan); clv=2*(x.close-x.low)/rng-1
 rows.append(pd.DataFrame({'date':x.index,'s':s,'rev':body,'clv':clv,'r':x.close.shift(-1)/x.close-1,'m5':ret.rolling(5).sum()}))
a=pd.concat(rows,ignore_index=True)
# regime is observable through t close and common to all assets: breadth of 5d returns
bread=a.groupby('date').m5.transform(lambda x:(x>0).mean())
# reversal is expected to work in stressed/broadly falling tape; retain smooth positive multiplier
# candidate: reverse signal times 1.5 when breadth<40%, 0.5 when breadth>60%
a['f']=a.rev*(1.5*(bread<.4)+1.0*((bread>=.4)&(bread<=.6))+.5*(bread>.6))
z=[]
for dt,g in a.dropna(subset=['f','r']).groupby('date'):
 if len(g)>=8:
  c=g.f.corr(g.r,method='spearman')
  if pd.notna(c): z.append((dt,c,len(g)))
z=pd.DataFrame(z,columns=['date','ic','n']).set_index('date'); print('factor conditional_stress_body_reversal cutoff',cut,'dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean(),'coverage',a.f.notna().mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=z.loc[lo:hi].ic; print('regime',lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
rank=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank'); print('turnover',rank.diff().abs().mean(axis=1).mean())
