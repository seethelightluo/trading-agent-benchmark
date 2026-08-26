import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 c=d['close'].astype(float); r=c.pct_change(); v=d['volume'].astype(float) if 'volume' in d else pd.Series(index=d.index,dtype=float)
 # Intermediate trend confirmed by abnormal volume, penalized only by downside volatility.
 trend=c.pct_change(40); abn=(v/(v.rolling(60,min_periods=30).median()+1e-12)).clip(.25,4)
 down=r.where(r<0,0).pow(2).rolling(40,min_periods=25).mean().pow(.5)*np.sqrt(40)
 sig=(trend*abn/(down+1e-8)).shift(1)
 P[s]=pd.DataFrame({'f':sig,'c':c})
rows=[]
for s,x in P.items():
 y=x.c.pct_change(10).shift(-10); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s; rows.append(z.reset_index())
a=pd.concat(rows,ignore_index=True)
def calc(h):
 vals=[]; ns=[]
 for dt,g in a.groupby('date'):
  yy=g.c.pct_change(h) if False else None
  # use aligned forward return from original prices
  parts=[]
  for s in g.s:
   pass
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   q=g.f.corr(g.y,method='spearman')
   if pd.notna(q): vals.append(q); ns.append(len(g))
 return vals,ns
ic,ns=calc(10); q=pd.Series(ic)
rank=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
print('assets',len(P),'dates',len(ic),'avgN',np.mean(ns),'coverage',len(a)/(len(set(a.date))*len(U)))
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',rank.diff().abs().mean(axis=1).mean()/2)
for h in [1,5,10,20]:
 vals=[]; ns2=[]
 for s,x in P.items():
  yy=x.c.pct_change(h).shift(-h); z=pd.concat([x.f,yy.rename('y')],axis=1).dropna(); z['s']=s
  for dt,g in z.reset_index().groupby('date'):
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
    qq=g.f.corr(g.y,method='spearman')
    if pd.notna(qq): vals.append(qq); ns2.append(len(g))
 qq=pd.Series(vals)
 print('decay',h,'dates',len(vals),'avgN',np.mean(ns2),'IC',qq.mean(),'ICIR',qq.mean()/qq.std(ddof=1))
a.to_csv('scripts/miner_2_20350412_downside_volume_trend_panel.csv',index=False)
pd.DataFrame({'ic':ic}).to_csv('scripts/miner_2_20350412_downside_volume_trend_ic.csv',index=False)
