import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().loc[:'2034-05-10']
r=np.log(C).diff(); cs=r.mean(axis=1); idio=r.sub(cs,axis=0)
# Candidate: longer residual reversal, normalized by downside risk and conditioned on persistent tail breadth.
raw=-idio.rolling(20,min_periods=12).sum().shift(1)
risk=idio.where(idio<0).rolling(60,min_periods=30).std().shift(1)
tail=(idio < -idio.rolling(60,min_periods=30).std().shift(1)).sum(axis=1)/C.notna().sum(axis=1)
bread=tail.rolling(10,min_periods=5).mean().shift(1)
f=(raw/risk.replace(0,np.nan)).mul((1+2*bread.clip(0,.8)).clip(1,2.6),axis=0)
f=f.rank(axis=1,pct=True).rolling(3,min_periods=2).mean(); f=f.sub(f.mean(axis=1),axis=0)
def forward(h): return np.log(C.shift(-h)/C)
def calc(x):
 out=[]; ns=[]; dates=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(d)
 return pd.Series(out,index=dates),pd.Series(ns,index=dates)
i,n=calc(forward(10)); print('candidate tail_persistence_reversal_20d'); print('dates',len(i),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,4),'start',i.index.min().date(),'end',i.index.max().date()); print('IC',round(i.mean(),6),'ICIR',round(i.mean()/i.std(ddof=1),6),'hit',round((i>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for w in [365,750,1260]:
 x=i.tail(w); print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for h in [1,5,20]: print('decay',h,round(calc(forward(h))[0].mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340511_tail_persistence_reversal_signal.csv',index=False)
i.rename('ic').to_csv('scripts/miner_1_20340511_tail_persistence_reversal_ic.csv')
