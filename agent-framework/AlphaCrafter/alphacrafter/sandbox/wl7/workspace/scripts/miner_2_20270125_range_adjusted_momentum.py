import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=3000) for s in U}
# Range-adjusted medium momentum: lagged 15d return divided by lagged mean true range / close.
rows=[]
for s,df in D.items():
 if df is None: continue
 df=df.copy(); df['date']=pd.to_datetime(df.date); df=df.set_index('date').sort_index()
 c=df.close.astype(float); h=df.high.astype(float); l=df.low.astype(float); prev=c.shift(1)
 tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
 atr=(tr/c).rolling(20).mean()
 sig=c.shift(1).pct_change(15)/atr.shift(1)
 fwd=c.shift(-1)/c-1
 for dt in sig.index:
  if pd.notna(sig.loc[dt]) and pd.notna(fwd.loc[dt]): rows.append((dt,s,float(sig.loc[dt]),float(fwd.loc[dt])))
x=pd.DataFrame(rows,columns=['date','sym','factor','fwd'])
ics=[]; turnovers=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>2 and g.fwd.nunique()>2:
  ics.append(g.factor.corr(g.fwd,method='spearman'))
# turnover rank changes, using consecutive common dates
wide=x.pivot(index='date',columns='sym',values='factor'); ranks=wide.rank(axis=1,pct=True)
turn=[]
for i in range(1,len(ranks)):
 a,b=ranks.iloc[i-1],ranks.iloc[i]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8: turn.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
ics=pd.Series(ics).dropna();
print('candidate=range_adjusted_momentum_15d dates',len(ics),'avg_n',x.groupby('date').size().mean(),'coverage',x.sym.nunique()/15,'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'turnover',np.mean(turn))
for H in [5,10,20]:
 vals=[]
 for s,df in D.items():
  if df is None: continue
  df=df.copy(); df.date=pd.to_datetime(df.date); df=df.set_index('date').sort_index(); c=df.close.astype(float); h=df.high.astype(float); l=df.low.astype(float); p=c.shift(1)
  tr=pd.concat([h-l,(h-p).abs(),(l-p).abs()],axis=1).max(axis=1); atr=(tr/c).rolling(20).mean(); sig=c.shift(1).pct_change(15)/atr.shift(1); fwd=c.shift(-H)/c-1
  vals += [(d,s,sig.loc[d],fwd.loc[d]) for d in sig.index if pd.notna(sig.loc[d]) and pd.notna(fwd.loc[d])]
 z=pd.DataFrame(vals,columns=['date','s','f','r']); q=[]
 for d,g in z.groupby('date'):
  if len(g)>=8:q.append(g.f.corr(g.r,method='spearman'))
 q=pd.Series(q).dropna(); print('H',H,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 q=[]
 for d,g in x.groupby('date'):
  if a<=d.year<=b and len(g)>=8:q.append(g.factor.corr(g.fwd,method='spearman'))
 q=pd.Series(q).dropna(); print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
