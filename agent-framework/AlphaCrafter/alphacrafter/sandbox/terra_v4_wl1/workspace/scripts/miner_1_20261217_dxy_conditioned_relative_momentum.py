import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym):
 d=pd.read_csv('../persistent/stock_data/'+sym+'.csv',parse_dates=['date']).set_index('date'); return d['close'].sort_index()
def calc():
 px=pd.concat({s:load(s) for s in U},axis=1).sort_index()
 dx=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 # completed-date signals: relative 20d momentum, conditioned by direction of DXY 20d trend
 r20=px.pct_change(20); d20=dx.pct_change(20).reindex(px.index).ffill()
 csmed=r20.median(axis=1)
 f=r20.sub(csmed,axis=0)*np.sign(d20).replace(0,np.nan).values[:,None]
 f=f.replace([np.inf,-np.inf],np.nan)
 fwd=px.pct_change().shift(-1)
 vals=[]; turns=[]
 for dt in f.index:
  a=f.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  if dt>f.index[0]:
   prev=f.shift(1).loc[dt]; q=pd.concat([a,prev],axis=1).dropna()
   if len(q)>=8: turns.append(np.mean(np.sign(q.iloc[:,0])!=np.sign(q.iloc[:,1])))
 v=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); x=v.ic
 print('dates',len(v),'avg_n',v.n.mean(),'coverage',v.n.sum()/(len(v)*15),'IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),np.mean(turns)))
 for h in [5,10]:
  fw=px.pct_change(h).shift(-h); q=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  q=pd.Series(q);print(h,'d',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
 v['year']=v.index.year;print(v.groupby('year').ic.mean().round(5).to_dict())
if __name__=='__main__':calc()
