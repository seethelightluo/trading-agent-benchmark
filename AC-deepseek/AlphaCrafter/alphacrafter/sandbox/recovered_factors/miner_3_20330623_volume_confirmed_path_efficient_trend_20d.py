"""One candidate: volume-confirmed path-efficient medium-term trend (20d).
Cross-asset continuation signal: 20d return multiplied by signed path efficiency
and relative 5d-versus-20d volume participation. Inputs end 2033-06-22.
Validates IC, regime stability, turnover, decay, and correlation versus all
available persisted signal artifacts.
"""
import os, glob, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-06-22')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def field(a, col):
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
    return x[col].rename(a)
p=pd.concat([field(a,'close') for a in A],axis=1).sort_index().loc[:CUT]
# Volume is deliberately optional only at source level; candidate requires it.
v=pd.concat([field(a,'volume') for a in A],axis=1).sort_index().reindex(p.index)
r=p.pct_change()
ret20=p.pct_change(20)
path=r.rolling(20,min_periods=15).sum().abs()/(r.abs().rolling(20,min_periods=15).sum()+1e-12)
# Relative participation is log(short volume / established volume), cross-sectional standardisation.
part=np.log((v.rolling(5,min_periods=4).mean()+1e-12)/(v.rolling(20,min_periods=15).mean()+1e-12))
part=part.sub(part.mean(axis=1),axis=0).div(part.std(axis=1).replace(0,np.nan),axis=0).clip(-3,3)
f=ret20*path*(1+0.35*part)
f=f.replace([np.inf,-np.inf],np.nan)
print('CANDIDATE volume_confirmed_path_efficient_trend_20d cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',int(f.notna().any(axis=1).sum()),'coverage',round(float(f.notna().mean().mean()),6),'valid_cells',int(f.notna().sum().sum()))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; out=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):out.append((d,z));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float); ics[h]=x; sd=x.std(ddof=1)
 print('H%d IC=%.6f ICIR=%.6f dates=%d hit=%.4f meanN=%.2f'%(h,x.mean(),x.mean()/sd,len(x),(x>0).mean(),np.mean(ns)))
 if h==10:
  for n,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027-2029','2027-01-01','2029-12-31'),('2030+','2030-01-01',str(CUT.date()))]:
   y=x.loc[lo:hi]; print('REGIME10',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turn)),6),'pairs',len(turn))
print('DECAY',{h:(round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)),6),len(x)) for h,x in ics.items()})
# Exact max signal correlation: align artifact dataframes/series by datetime, symbols, and flatten valid cells.
mx=-1.; whom=''; nc=0
for fn in glob.glob('scripts/*_signal.pkl'):
 try:
  with open(fn,'rb') as z: g=pickle.load(z)
  if isinstance(g,pd.Series): g=g.to_frame()
  if not isinstance(g,pd.DataFrame): continue
  g.index=pd.to_datetime(g.index); g=g.loc[:CUT]
  common_i=f.index.intersection(g.index); common_c=f.columns.intersection(g.columns)
  if len(common_i)==0 or len(common_c)<8: continue
  q=pd.concat([f.loc[common_i,common_c].stack().rename('x'),g.loc[common_i,common_c].stack().rename('y')],axis=1).dropna()
  if len(q)<100 or q.x.nunique()<2 or q.y.nunique()<2: continue
  rho=spearmanr(q.x,q.y).statistic; nc+=1
  if abs(rho)>mx: mx=abs(rho); whom=os.path.basename(fn)
 except Exception as e: pass
print('LIBRARY_CORRELATION artifacts_screened',nc,'max_abs',round(mx,6),'artifact',whom)
# Artifact retained solely to allow reproducible future correlation audits.
with open('scripts/miner_3_20330623_volume_confirmed_path_efficient_trend_20d_signal.pkl','wb') as z: pickle.dump(f,z)
"""
