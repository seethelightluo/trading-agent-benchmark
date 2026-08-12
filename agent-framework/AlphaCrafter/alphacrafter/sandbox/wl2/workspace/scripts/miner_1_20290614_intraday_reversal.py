import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# candidate: idiosyncratic prior-session intraday return reversal, volatility scaled
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
 if len(d):
  d=d.drop_duplicates('date').set_index('date').sort_index()
  D[s]=d
# common dates, use close and open; factor at date t uses prior row's intraday return, lagged
frames=[]
for s,d in D.items():
 x=pd.DataFrame(index=d.index)
 x['r']=d['close'].pct_change()
 x['intra']=d['close']/d['open']-1
 x['vol']=d['close'].pct_change().rolling(20).std()
 x['fraw']=x['intra'].shift(1) # available after prior completed day
 x['f']=(-x['fraw'])/(x['vol'].shift(1)+1e-6)
 x['y']=x['r']
 x['s']=s; frames.append(x.reset_index())
a=pd.concat(frames).rename(columns={'index':'date'}).dropna(subset=['f','y'])
# cross sectional IC each date, require 8
ics=[]; ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:
  ics.append(g['f'].corr(g['y'],method='spearman')); ns.append(len(g))
z=pd.Series(ics).dropna(); print('dates',len(z),'assets',len(U),'avgN',np.mean(ns),'coverage',len(a)/sum(len(d) for d in D.values()))
print('IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit', (z>0).mean(),'turnover_proxy',a.groupby('s')['f'].apply(lambda x:(x.diff().abs()>0.5).mean()).mean())
for h in [1,3,5,10]:
 # forward h-day return from date t close to t+h close, signal at t
 vals=[]
 for s,d in D.items():
  q=pd.DataFrame(index=d.index); q['f']=(-(d['close']/d['open']-1).shift(1))/(d['close'].pct_change().rolling(20).std().shift(1)+1e-6); q['y']=d['close'].shift(-h)/d['close']-1; q['s']=s; vals.append(q.reset_index())
 b=pd.concat(vals).dropna(); zz=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8: zz.append(g['f'].corr(g['y'],method='spearman'))
 zz=pd.Series(zz).dropna(); print('h',h,'dates',len(zz),'IC',zz.mean(),'ICIR',zz.mean()/zz.std(ddof=1),'hit',(zz>0).mean())
# save artifact
out=a[['date','s','f']].rename(columns={'s':'symbol','f':'signal'}); out.to_csv('scripts/miner_1_20290614_intraday_reversal_signal.csv',index=False)
