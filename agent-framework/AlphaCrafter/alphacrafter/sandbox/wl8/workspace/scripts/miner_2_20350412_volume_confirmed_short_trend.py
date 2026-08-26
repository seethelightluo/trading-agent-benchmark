import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); c=d.close.astype(float); r=c.pct_change(); v=d.volume.astype(float)
 tr=c.pct_change(20); ab=(v/(v.rolling(40,min_periods=20).median()+1e-12)).clip(.25,4); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
 P[s]=pd.DataFrame({'f':(tr*ab/(vol+1e-12)).shift(1),'c':c})

def run(h):
 out=[]; ns=[]
 for s,x in P.items():
  y=x.c.pct_change(h).shift(-h); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s
  for dt,g in z.reset_index().groupby('date'):
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
    q=g.f.corr(g.y,method='spearman')
    if pd.notna(q): out.append(q); ns.append(len(g))
 q=pd.Series(out); return len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('assets',len(P),'coverage',sum(x.f.notna().sum() for x in P.values())/(len(P)*max(len(x) for x in P.values())))
for h in [1,5,10,20]: print('horizon',h,'dates avgN IC ICIR hit',run(h))
# artifacts at admission horizon
rows=[]
for s,x in P.items():
 y=x.c.pct_change(10).shift(-10); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s; rows.append(z.reset_index())
a=pd.concat(rows); a.to_csv('scripts/miner_2_20350412_volume_confirmed_short_trend_panel.csv',index=False)
