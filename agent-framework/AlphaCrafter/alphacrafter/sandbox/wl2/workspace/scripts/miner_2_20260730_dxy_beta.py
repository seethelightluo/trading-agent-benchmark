import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff='2026-07-15'
def load(path):
 d=pd.read_csv(path); d['date']=pd.to_datetime(d['date']); d=d[d.date<=cutoff].set_index('date'); return d['close'].astype(float)
px=pd.concat({a:load('../persistent/stock_data/'+a+'.csv') for a in assets},axis=1).sort_index(); dxy=load('../persistent/index_data/DXY.csv').reindex(px.index).ffill(); r=px.pct_change(); dr=dxy.pct_change()
cov=r.rolling(60,min_periods=45).cov(dr); var=dr.rolling(60,min_periods=45).var(); f=-cov.div(var,axis=0).shift(1)
fr=[]; dates=[]; ns=[]
for i in range(len(px)-1):
 z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: fr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(px.index[i]); ns.append(len(z))
s=pd.Series(fr,index=pd.DatetimeIndex(dates)).dropna(); print('DXY beta defensive | dates',len(s),'avg names',np.mean(ns),'coverage',np.mean(ns)/15); print('daily IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
for h in [5,10]:
 vals=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],px.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(vals).dropna(); print('%dd IC %.6f ICIR %.6f dates %d'%(h,q.mean(),q.mean()/q.std(),len(q)))
for y in range(2020,2027):
 q=s[s.index.year==y]; print(y,len(q),round(q.mean(),5),round(q.mean()/q.std(),4) if len(q)>2 else None)
print('rank turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for name,ff in [('mom',r.rolling(20).mean().div(r.rolling(20).std()).shift(1)),('rev',-px.pct_change(5).shift(1)),('clv',(px-px.rolling(20).min()).div(px.rolling(20).max()-px.rolling(20).min()).shift(1))]:
 z=pd.concat([f.stack(),ff.stack()],axis=1).dropna(); print('corr',name,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
