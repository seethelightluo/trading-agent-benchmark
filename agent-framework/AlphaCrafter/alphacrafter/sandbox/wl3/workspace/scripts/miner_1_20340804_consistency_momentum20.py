import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=6000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill()
# consistency-weighted momentum: lagged 20d return times fraction of up days, volatility scaled
r=p.pct_change()
ret20=p.shift(1)/p.shift(21)-1
cons=r.shift(1).rolling(20,min_periods=18).apply(lambda z: np.mean(z>0),raw=True)
vol=r.shift(1).rolling(20,min_periods=18).std()*np.sqrt(252)
f=ret20*cons/(vol+1e-8)
# forward 10 sessions from decision date
fr=p.shift(-10)/p-1
rows=[]
for dt in f.index:
    a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
ic=pd.Series({d:v for d,n,v in rows})
print('dates',len(ic),'avg_n',np.mean([n for d,n,v in rows]),'coverage',len(ic)/len(f))
print('IC %.8f ICIR %.8f hit %.4f'%(ic.mean(),ic.mean()/(ic.std(ddof=1)+1e-12), (ic>0).mean()))
for w in [120,252,756]:
 q=ic.tail(w); print('recent',w,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/(q.std(ddof=1)+1e-12),'hit',(q>0).mean())
# turnover rank signal, mean daily rank changes
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('turnover_rank_change',turn,'last',f.iloc[-1].round(4).to_dict())
print('decay',[(h, pd.concat([f,p.shift(-h)/p-1],axis=1).dropna(axis=0).groupby(level=0).apply(lambda q:q.iloc[:,:len(p.columns)].iloc[0].corr(q.iloc[:,len(p.columns):].iloc[0]) if False else np.nan)) for h in []])
# decay with same cross-sectional loop
for h in [1,5,10,20]:
 z=[]
 ff=f; yy=p.shift(-h)/p-1
 for d in ff.index:
  q=pd.concat([ff.loc[d],yy.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(z),len(z))
