import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; frames=[]; raw={}
for s in U:
 d=pd.read_csv(base+'/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index(); raw[s]=d
 rng=(d.high-d.low).replace(0,np.nan)
 f=-(0.6*((d.close-d.low)/rng-0.5)*2+0.4*(d.close-d.open)/rng).rolling(5,min_periods=5).mean()
 frames.append(pd.DataFrame({'date':d.index,'symbol':s,'f':f,'ret':d.close.pct_change().shift(-1)}))
x=pd.concat(frames,ignore_index=True).dropna(); rows=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.ret.nunique()>1: rows.append((dt,g.f.corr(g.ret,method='spearman'),len(g)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(r),'meanN',r.n.mean(),'assets',x.symbol.nunique()); print('meanIC %.6f std %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.std(ddof=1),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean()))
for h in [1,5,10]:
 z=[]
 for s,d in raw.items():
  rng=(d.high-d.low).replace(0,np.nan); f=-(0.6*((d.close-d.low)/rng-0.5)*2+0.4*(d.close-d.open)/rng).rolling(5,min_periods=5).mean(); rr=d.close.pct_change(h).shift(-h); z.append(pd.DataFrame({'date':d.index,'f':f,'ret':rr,'symbol':s}))
 z=pd.concat(z,ignore_index=True).dropna(); q=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.ret.nunique()>1:q.append(g.f.corr(g.ret,method='spearman'))
 q=pd.Series(q).dropna(); print('h',h,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
wide=x.pivot_table(index='date',columns='symbol',values='f'); turn=wide.rank(pct=True,axis=1).diff().abs().mean(axis=1).dropna(); print('turnover',turn.mean())
