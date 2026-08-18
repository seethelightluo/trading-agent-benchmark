import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv')
 p['date']=pd.to_datetime(p['date']); p=p.sort_values('date').set_index('date')
 D[s]=p['close'].astype(float)
px=pd.concat(D,axis=1).sort_index().ffill()
r=np.log(px).diff()
# Candidate: downside-resilient trend: 30d return normalized by downside semivolatility,
# with upside/downside balance; all inputs lagged one day.
down=r.clip(upper=0).rolling(30,min_periods=20).std()
up=r.clip(lower=0).rolling(30,min_periods=20).mean()
dn=(-r.clip(upper=0)).rolling(30,min_periods=20).mean()
trend=np.log(px).diff(30)
f=(trend/(down*np.sqrt(30)+1e-8)) + 0.5*np.log((up+1e-5)/(dn+1e-5))
f=f.shift(1)
fr=np.log(px).shift(-10)-np.log(px)
rows=[]
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# dates after enough warmup and exclude terminal missing
x=x.loc['2024-01-01':'2033-08-20']
print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover_proxy %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean(), f.diff().rank(axis=1).corr(f.rank(axis=1).shift(1),method='spearman').stack().mean() if False else np.nan))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=x.loc[a:b]; print(a,b,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(),6),round((q.ic>0).mean(),4))
# rank turnover
rr=f.rank(axis=1,pct=True); t=(rr.diff().abs().mean(axis=1)).dropna(); print('rank turnover',t.loc[x.index].mean())
out=pd.DataFrame(f.stack(),columns=['signal']); out.index.names=['date','symbol']; out.reset_index().to_csv('scripts/miner_3_20330902_tail_resilient_signal.csv',index=False)
