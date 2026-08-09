import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut] for a in assets}; v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].loc[:cut]
rets=pd.DataFrame({a:s.pct_change() for a,s in px.items()}); common=rets.join(v.pct_change().rename('vix'),how='inner').sort_index(); beta=common[assets].rolling(60,min_periods=50).cov(common.vix).div(common.vix.rolling(60,min_periods=50).var(),axis=0); factor=-beta
fwd=pd.DataFrame({a:px[a].pct_change().shift(-1) for a in assets}).reindex(common.index)
def calc(f):
 q=[];ix=[];nn=[]
 for d in common.index:
  z=pd.concat([factor.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ix.append(d);nn.append(len(z))
 return pd.Series(q,index=ix),nn
ics,nn=calc(fwd); print('cutoff',cut.date(),'dates',len(ics),'mean_names',np.mean(nn),'coverage',factor.stack().notna().mean());print('1d IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),(ics>0).mean()));print('years',ics.groupby(ics.index.year).mean().round(5).to_dict())
for h in [5,10]:
 f=pd.DataFrame({a:px[a].pct_change(h).shift(-h) for a in assets}).reindex(common.index);q,n=calc(f);print(h,'IC %.6f ICIR %.6f N %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
r=factor.rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).mean())
for name,s in [('reversal',-rets.rolling(5).sum()),('mom',rets.rolling(20).sum()),('ram',rets.rolling(20).sum()/rets.rolling(20).std())]:
 z=pd.concat([factor.stack().rename('f'),s.stack().rename('x')],axis=1).dropna();print('corr',name,z.f.corr(z.x))
