import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-03-15')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date').close.astype(float) for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); r20=p.pct_change(20); vol=r.rolling(20).std()*np.sqrt(20); breadth=r.gt(0).rolling(20).mean(); down=r.clip(upper=0).abs().rolling(20).mean(); denom=r.rolling(20).mean().abs()+1e-12
sig=(r20/vol)*(0.5+0.5*breadth)/(1+down/denom); fwd=p.shift(-10)/p-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(out),'avg_names',round(out.n.mean(),3),'coverage',round(out.n.sum()/(len(out)*15),6)); print('IC %.8f ICIR %.8f hit %.6f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1),np.mean(out.ic>0)))
for h in [5,10,20,40]:
 ff=p.shift(-h)/p; q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(float(np.mean(q)),8),len(q))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]:
 q=out.loc[a:b]; print('regime',a,b,len(q),round(q.ic.mean(),8),round(q.ic.mean()/q.ic.std(ddof=1),6),round((q.ic>0).mean(),4))
print('turnover',round(float((sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())),6)); sig.tail(1).T.to_csv('scripts/miner_3_20340316_upside_efficiency_momentum_20d_signal.csv')
