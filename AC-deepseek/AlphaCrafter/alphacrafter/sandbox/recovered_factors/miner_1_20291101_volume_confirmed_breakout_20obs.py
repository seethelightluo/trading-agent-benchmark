import pandas as pd,numpy as np,json,glob
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-10-31')
def rd(a,col,n=5000):
 d=get_stock_daily_data(a,n).set_index('date'); d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,col],errors='coerce')
p=pd.DataFrame({a:rd(a,'close') for a in A}); v=pd.DataFrame({a:rd(a,'volume') for a in A})
r=p.pct_change()
# Volume-confirmed breakout: medium return, scaled by unusually high participation; lag one day.
ret=p/p.shift(20)-1
rv=v/v.rolling(60,min_periods=40).median()
f=(ret*rv.clip(upper=4)).shift(1)
# cross-sectional residualization removes broad risk/volatility exposure
vol=r.rolling(20,min_periods=15).std().shift(1)
for d in f.index:
 z=pd.concat([f.loc[d].rename('f'),vol.loc[d].rename('v')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['v']].to_numpy()]
  f.loc[d,z.index]=z.f-X@np.linalg.lstsq(X,z.f,rcond=None)[0]
def calc(h):
 fw=p.shift(-h)/p-1; vals=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
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
# Full library screen: pooled date-asset Spearman against every JSON expression proxy unavailable; report exact controls and library count.
for path in glob.glob('factors/*.json'):
 try:
  q=json.load(open(path)); mid=q.get('factor_id','')
  if q.get('validation',{}).get('status')=='EFFECTIVE': print('LIBRARY_FACTOR',mid)
 except: pass
for n,x in [('ret20',ret),('vol',vol),('rv',rv)]:
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).replace([np.inf,-np.inf],np.nan).dropna(); print('CONTROL_RHO',n,z.f.corr(z.x,method='spearman'),len(z))
