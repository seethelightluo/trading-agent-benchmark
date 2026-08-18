import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut] for a in assets}; dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]
rets=pd.DataFrame({a:s.pct_change() for a,s in px.items()}); common=rets.join(dxy.pct_change().rename('dxy'),how='inner').sort_index()
fwd=pd.DataFrame({a:px[a].pct_change().shift(-1) for a in assets}).reindex(common.index)
for w in [40,60,90]:
 beta=common[assets].rolling(w,min_periods=max(30,w-10)).cov(common.dxy).div(common.dxy.rolling(w,min_periods=max(30,w-10)).var(),axis=0); factor=-beta
 def calc(y):
  q=[]; ds=[]; nn=[]
  for d in common.index:
   z=pd.concat([factor.loc[d],y.loc[d]],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);nn.append(len(z))
  return pd.Series(q,index=ds),nn
 ic,nn=calc(fwd); print('window',w,'dates',len(ic),'mean_names',round(np.mean(nn),2),'coverage',round(factor.stack().notna().mean(),4),'IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean()))
 for h in [5,10]:
  y=pd.DataFrame({a:px[a].pct_change(h).shift(-h) for a in assets}).reindex(common.index);q,n=calc(y);print(' horizon',h,'IC %.6f ICIR %.6f dates %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
 print('regime',ic.groupby(ic.index.year).mean().round(4).to_dict(),'turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
 for name,s in [('rev',-rets.rolling(5).sum()),('mom',rets.rolling(20).sum())]:
  z=pd.concat([factor.stack().rename('f'),s.stack().rename('x')],axis=1).dropna(); print(' corr',name,round(z.f.corr(z.x),4))
