import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
r={h:P.shift(1).div(P.shift(h+1))-1 for h in [3,5,10,20,40,60]}
# lagged signals: short reversal plus longer trend; cross-sectional z scores

def z(x):
 m=x.mean(axis=1); sd=x.std(axis=1).replace(0,np.nan); return x.sub(m,axis=0).div(sd,axis=0)
def evaluate(name,f):
 out=[]
 for h in [1,5,10]:
  y=P.shift(-h).div(P)-1; rows=[]
  for d in P.index:
   q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
   if len(q)>=8: rows.append(q.f.corr(q.y))
  a=pd.Series(rows).dropna(); out.append((h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 print(name, out)
for w in [.15,.25,.35,.5,.75,1.0]:
 f=z(-r[3])+w*z(r[60]); evaluate('shortrev+%g*trend60'%w,f)
f=z(-r[3])+0.35*z(r[40]); evaluate('shortrev+0.35trend40',f)
print('period',P.index.min(),P.index.max(),'dates',len(P))
