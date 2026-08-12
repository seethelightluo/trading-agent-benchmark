import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-07-15')
def load(path):
 d=pd.read_csv(path); d['date']=pd.to_datetime(d['date']); d=d[d['date']<=cutoff].drop_duplicates('date').set_index('date'); return d['close'].astype(float)
px=pd.concat([load('../persistent/stock_data/'+a+'.csv').rename(a) for a in assets],axis=1).sort_index(); px.index=pd.DatetimeIndex(px.index)
dxy=load('../persistent/index_data/DXY.csv').reindex(px.index).ffill(); r=px.pct_change(); dr=dxy.pct_change()
mu_r=r.rolling(60,min_periods=45).mean(); mu_d=dr.rolling(60,min_periods=45).mean(); cov=(r.mul(dr,axis=0).rolling(60,min_periods=45).mean()-mu_r.mul(mu_d,axis=0)); var=dr.rolling(60,min_periods=45).var(); f=-(cov.div(var,axis=0)).shift(1)
def eval_h(h):
 out=[]; ns=[]; dates=[]; fr=px.shift(-h).div(px)-1
 for dt in px.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 return pd.Series(out,index=pd.DatetimeIndex(dates)).dropna(),ns
s,ns=eval_h(1); print('DXY beta defensive | cutoff',cutoff.date(),'last_px',px.index.max().date(),'dates',len(s),'avg names',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4)); print('daily IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
for h in [5,10]:
 q,n=eval_h(h); print('%dd IC %.6f ICIR %.6f dates %d'%(h,q.mean(),q.mean()/q.std(),len(q)))
for y in range(2020,2027):
 q=s[s.index.year==y]; print(y,len(q),round(q.mean(),5),round(q.mean()/q.std(),4) if len(q)>2 else None)
print('rank turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for name,ff in [('mom',r.rolling(20).mean().div(r.rolling(20).std()).shift(1)),('rev',-px.pct_change(5).shift(1)),('clv',(px-px.rolling(20).min()).div(px.rolling(20).max()-px.rolling(20).min()).shift(1))]:
 z=pd.concat([f.stack(),ff.stack()],axis=1).dropna(); print('corr',name,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
