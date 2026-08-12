import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-12-26'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
 D[s]=x.set_index('date').close
px=pd.DataFrame(D).sort_index().loc[:end].astype(float); ret=px.pct_change(); rows=[]
for i in range(100,len(px)-21):
 date=px.index[i]; r40=px.iloc[i-1]/px.iloc[i-41]-1; vol=ret.iloc[i-61:i-1].std();
 mkt=ret.iloc[:i].mean(axis=1); beta=pd.Series(index=px.columns,dtype=float)
 for s in px.columns:
  z=pd.concat([ret[s].iloc[i-21:i-1],mkt.iloc[i-21:i-1]],axis=1).dropna()
  beta[s]=z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,1].var() if len(z)>8 else np.nan
 f=(r40-r40.median())/vol.replace(0,np.nan)
 q=pd.DataFrame({'f':f,'vol':vol,'beta':beta}).dropna()
 if len(q)<8: continue
 X=np.column_stack([np.ones(len(q)),q.vol.rank().values,q.beta.rank().values]); coef=np.linalg.lstsq(X,q.f.values,rcond=None)[0]; sig=q.f-(X@coef)
 for h in [1,5,10,20]:
  fr=px.iloc[i+h]/px.iloc[i]-1; z=pd.concat([sig,fr.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((date,h,z.iloc[:,0].corr(z.y,method='spearman'),len(z)))
df=pd.DataFrame(rows,columns=['date','h','ic','n']); print('dates',df.date.nunique(),'obs',len(df),'avgN',df.groupby('date').n.mean().mean())
for h,g in df.groupby('h'):
 print(h,'IC',g.ic.mean(),'ICIR',g.ic.mean()/g.ic.std(ddof=1),'hit',(g.ic>0).mean(),'n',len(g))
g=df[df.h==10].copy();g['yr']=pd.to_datetime(g.date).dt.year; print(g.groupby('yr').ic.agg(['mean','count']).to_string())
