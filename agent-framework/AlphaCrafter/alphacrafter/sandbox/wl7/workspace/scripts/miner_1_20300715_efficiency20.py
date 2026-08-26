import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d): frames[s]=pd.Series(d.close.astype(float).values,index=pd.to_datetime(d.date)).groupby(level=0).last()
px=pd.DataFrame(frames).sort_index().ffill(); r=px.pct_change()
# Lagged trend efficiency: directional 20d return divided by total absolute path movement.
factor=(px.pct_change(20)/(r.abs().rolling(20).sum()+1e-8)).shift(1)
def calc(h):
 rows=[]
 fwd=px.shift(-h)/px-1
 for dt in px.index:
  z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
res=calc(10)
for lab,rr in [('full',res),('early',res.iloc[:len(res)//3]),('mid',res.iloc[len(res)//3:2*len(res)//3]),('late',res.iloc[2*len(res)//3:])]:
 m=rr.ic.mean(); sd=rr.ic.std(ddof=1); print(lab,'dates',len(rr),'avg_n',round(rr.n.mean(),2),'IC',round(m,8),'ICIR',round(m/sd*np.sqrt(252),8),'hit',round((rr.ic>0).mean(),4))
for h in [1,5,10,20,40]:
 q=calc(h); print('decay',h,'IC',round(q.ic.mean(),8),'dates',len(q))
rank=factor.rank(axis=1,pct=True)
print('cutoff',px.index[-1],'assets',len(frames),'dates',len(px),'coverage',round(factor.notna().sum().sum()/(factor.shape[0]*len(U)),6),'turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
res.to_csv('scripts/miner_1_20300715_efficiency20_ic.csv'); factor.to_csv('scripts/miner_1_20300715_efficiency20_signal.csv')
