import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2033-08-03')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill().loc[:cutoff]
r=p.pct_change(); vol=r.rolling(20).std()*np.sqrt(252)
base=(p.shift(1)/p.shift(21)-1)/vol.shift(1); base=base.sub(base.median(axis=1),axis=0).rolling(10,min_periods=10).mean()
breadth=(p.shift(1)/p.shift(21)-1).gt(0).mean(axis=1); weak=breadth.rolling(5,min_periods=5).mean()<0.40
f=base.copy(); f.loc[weak]=(-base.loc[weak]); f.to_csv('scripts/miner_2_20330804_breadth_flip_volnorm_signal.csv')
fwd=p.shift(-10)/p-1; rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
ic=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
for label,q in [('full',ic),('recent365',ic.tail(365)),('recent730',ic.tail(730)),('pre_recent',ic.iloc[:-365])]:
 m=q.ic.mean(); sd=q.ic.std(ddof=1); print(label,'dates',len(q),'avg_n',q.n.mean(),'IC',m,'ICIR',m/sd if sd else np.nan,'hit',(q.ic>0).mean())
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().stack().mean(),'turnover',rank.diff().abs().mean(axis=1).mean()); print('period',ic.index.min(),ic.index.max())
