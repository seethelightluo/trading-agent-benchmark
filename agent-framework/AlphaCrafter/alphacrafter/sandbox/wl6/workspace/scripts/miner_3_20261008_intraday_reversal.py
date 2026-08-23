import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; S={}
for a in assets:
 p=f'{base}/{a}.csv'
 if not os.path.exists(p): continue
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 # fade completed session's open-to-close move, strictly known at decision after close
 intraday=d.close/d.open-1
 S[a]=pd.DataFrame({'f':-intraday,'c':d.close})
for h in [1,5,10]:
 rows=[]
 for a,x in S.items():
  q=x.copy(); q['r']=q.c.shift(-h)/q.c-1; q=q.dropna()
  rows += [(dt,a,f,r) for dt,f,r in zip(q.index,q.f,q.r)]
 d=pd.DataFrame(rows,columns=['date','a','f','r']); vals=[]; ns=[]
 for dt,g in d.groupby('date'):
  z=g[['f','r']].replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   vals.append(spearmanr(z.f,z.r).statistic); ns.append(len(z))
 v=np.array(vals); print(h,'dates',len(v),'avg_n',np.mean(ns),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0))
# turnover: daily rank ordering changes
allx=[]
for a,x in S.items():
 for dt,f in x.f.items(): allx.append((dt,a,f))
z=pd.DataFrame(allx,columns=['date','a','f']).pivot(index='date',columns='a',values='f').rank(axis=1,pct=True)
print('assets',len(S),'coverage',z.notna().stack().mean(),'turnover',z.diff().abs().mean().mean())
