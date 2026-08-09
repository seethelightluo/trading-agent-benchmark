import pandas as pd, numpy as np, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# orthogonal 20d momentum: own 20d return less cross-asset median 20d return, risk scaled
m=p.pct_change(20); vol=r.rolling(20).std(); fac=(m.sub(m.median(axis=1),axis=0)/vol).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(p)-1):
 x=fac.iloc[i]; y=r.iloc[i+1]
 z=pd.concat([x,y],axis=1).dropna();
 if len(z)>=8: rows.append((p.index[i],len(z),z.iloc[:,0].corr(z.iloc[:,1]),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
d=pd.DataFrame(rows,columns=['date','n','ic','ric']).set_index('date')
for col in ['ic','ric']:
 a=d[col].dropna(); print(col,'mean',a.mean(),'std',a.std(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'dates',len(a),'avgN',d.n.mean())
 for h in [1,5,10]:
  # same factor t vs forward compounded h-day
  out=[]
  for i in range(len(p)-h):
   z=pd.concat([fac.iloc[i],p.pct_change(h).iloc[i+h]],axis=1).dropna()
   if len(z)>=8: out.append(z.iloc[:,0].corr(z.iloc[:,1]))
  a=pd.Series(out).dropna(); print('h',h,'IC',a.mean(),'ICIR',a.mean()/a.std(),'n',len(a))
print('coverage',fac.notna().sum(axis=1).mean()/15)
# turnover mean rank signal changes
rr=fac.rank(axis=1,pct=True); print('turnover',rr.diff().abs().mean(axis=1).mean())
# artifact
fac.to_csv('scripts/miner_2_20270325_orthogonal_momentum_signal.csv',index_label='date')
