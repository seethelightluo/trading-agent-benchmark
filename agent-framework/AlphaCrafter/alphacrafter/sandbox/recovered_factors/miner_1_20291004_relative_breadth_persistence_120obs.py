import pandas as pd,numpy as np,json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-10-03')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index); return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A}); r=p.pct_change(); peer=r.mean(axis=1)
# one interpretable idea: persistent relative participation, residualized against trend and risk
breadth=(r.gt(peer,axis=0).astype(float)).rolling(120,min_periods=80).mean()
breadth=breadth.rolling(20,min_periods=10).mean()
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(60)-1)/vol; risk=-vol
f=breadth*np.nan
for d in p.index:
 z=pd.concat([breadth.loc[d].rename('y'),trend.loc[d].rename('t'),risk.loc[d].rename('v')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['t','v']].to_numpy()]; f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
 x=pd.Series(dict(vals)); o={'horizon':h,'ic':x.mean(),'icir':x.mean()/x.std(),'hit_ratio':(x>0).mean(),'dates':len(x),'mean_instruments':np.mean(ns)}
 for n,m in [('2020-25',x.index.year<=2025),('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029',x.index.year==2029),('latest120',np.arange(len(x))>=len(x)-120)]:
  q=x[m]; o[n]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std() if len(q)>1 else np.nan}
 return o
print('VISIBLE',END.date(),'assets',len(A),'rows',len(p),'signal_cells',int(f.notna().sum().sum()),'possible',f.size,'coverage',float(f.notna().sum().sum()/f.size))
for h in [1,5,10,20]: print('METRIC',json.dumps(calc(h),default=float))
turn=[]
for i in range(10,len(f),10):
 z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('TURNOVER_10D',np.mean(turn),'dates',len(turn))
# candidate-to-controls aligned Spearman evidence
for n,x in [('trend',trend),('risk',risk)]:
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); print('CONTROL_RHO',n,z.f.corr(z.x,method='spearman'),len(z))
