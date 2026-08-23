import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={s:get_stock_daily_data(s,days=3000) for s in U}; rows=[]
for s,df in D.items():
 if df is None:continue
 df=df.copy();df.date=pd.to_datetime(df.date);df=df.set_index('date').sort_index();c=df.close.astype(float);h=df.high.astype(float);l=df.low.astype(float);p=c.shift(1);tr=pd.concat([h-l,(h-p).abs(),(l-p).abs()],axis=1).max(axis=1);atr=(tr/c).rolling(20).mean();sig=-c.shift(1).pct_change(3)/atr.shift(1);fwd=c.shift(-1)/c-1
 for d in sig.index:
  if pd.notna(sig.loc[d]) and pd.notna(fwd.loc[d]):rows.append((d,s,sig.loc[d],fwd.loc[d]))
x=pd.DataFrame(rows,columns=['date','sym','factor','fwd']);q=[]
for d,g in x.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>2:q.append(g.factor.corr(g.fwd,method='spearman'))
q=pd.Series(q).dropna();w=x.pivot(index='date',columns='sym',values='factor').rank(axis=1,pct=True);t=[]
for i in range(1,len(w)):
 z=pd.concat([w.iloc[i-1],w.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('candidate=range_adjusted_reversal_3d dates',len(q),'avg_n',x.groupby('date').size().mean(),'coverage',x.sym.nunique()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',np.mean(t))
for H in [5,10,20]:
 z=[]
 for s,df in D.items():
  if df is None:continue
  df=df.copy();df.date=pd.to_datetime(df.date);df=df.set_index('date').sort_index();c=df.close.astype(float);h=df.high.astype(float);l=df.low.astype(float);p=c.shift(1);tr=pd.concat([h-l,(h-p).abs(),(l-p).abs()],axis=1).max(axis=1);a=(tr/c).rolling(20).mean();f=-c.shift(1).pct_change(3)/a.shift(1);r=c.shift(-H)/c-1;z += [(d,f.loc[d],r.loc[d]) for d in f.index if pd.notna(f.loc[d]) and pd.notna(r.loc[d])]
 y=pd.DataFrame(z,columns=['d','f','r']);v=[]
 for d,g in y.groupby('d'):
  if len(g)>=8:v.append(g.f.corr(g.r,method='spearman'))
 v=pd.Series(v).dropna();print('H',H,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
