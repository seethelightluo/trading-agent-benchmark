import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100: d=get_index_daily_data(s,5000)
 if d is not None: frames[s]=d[['date','close']].drop_duplicates('date').set_index('date')['close']
p=pd.DataFrame(frames).sort_index(); r=p.pct_change(); mom=r.rolling(20).sum(); vol=r.rolling(20).std(); base=mom/vol.replace(0,np.nan)
breadth=(mom>0).sum(axis=1)/mom.notna().sum(axis=1); reg=(2*breadth-1).shift(1); f=base.shift(1).mul(reg,axis=0).replace([np.inf,-np.inf],np.nan); fwd=p.shift(-10)/p-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z),breadth.loc[dt]))
out=pd.DataFrame(rows,columns=['date','ic','n','breadth']).set_index('date')
print('dates',len(out),'range',out.index.min(),out.index.max(),'avg_n',out.n.mean())
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f' % (out.ic.mean(),out.ic.mean()/out.ic.std(),(out.ic>0).mean(),f.notna().sum().sum()/(len(f)*len(U)),f.rank(axis=1).diff().abs().sum().sum()/(len(f)*len(U))))
for n in [120,260,520,780]:
 q=out.tail(n).ic; print('recent',n,'IC %.6f ICIR %.6f hit %.4f dates %d'%(q.mean(),q.mean()/q.std(),(q>0).mean(),len(q)))
for label,mask in [('lowbreadth',out.breadth<.4),('mid',(out.breadth>=.4)&(out.breadth<=.6)),('highbreadth',out.breadth>.6)]:
 q=out.loc[mask,'ic']; print(label,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out.to_csv('scripts/artifacts/miner_2_20341109_breadth_regime_momentum_ic.csv'); f.to_csv('scripts/artifacts/miner_2_20341109_breadth_regime_momentum_signal.csv')
