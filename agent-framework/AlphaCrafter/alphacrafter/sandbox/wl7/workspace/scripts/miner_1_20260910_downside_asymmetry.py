import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,2400)
   if x is not None and len(x)>150:return x
  except: pass
D={}
for s in U:
 x=fetch(s)
 if x is not None:
  x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); D[s]=x.drop_duplicates('date').set_index('date').sort_index()
# downside asymmetry: favor assets with positive/low downside capture, continuously ranked
F={};R={}
for s,d in D.items():
 r=d.close.astype(float).pct_change(); dn=r.where(r<0,0.0)
 down=dn.abs().rolling(30,min_periods=20).mean(); total=r.abs().rolling(30,min_periods=20).mean()
 # high upside participation and low downside intensity, stabilized by total movement
 F[s]=((r.rolling(30,min_periods=20).mean()/total) - down/total).shift(1)
 R[s]=r
all_dates=sorted(set().union(*[set(x.index) for x in F.values()])); ic=[]; obs=[]; hs={5:[],10:[]}; regs={}; prev=None; turns=[]
for dt in all_dates:
 vals={s:F[s].loc[dt] for s in F if dt in F[s].index and np.isfinite(F[s].loc[dt])}
 out={}
 for s,v in vals.items():
  ix=R[s].index; p=ix.get_loc(dt)
  out[s]={k:R[s].iloc[p+1:p+k+1].sum() for k in (1,5,10) if p+k<len(ix)}
 z=[(v,out[s][1]) for s,v in vals.items() if s in out and 1 in out[s]]
 if len(z)>=8:
  q=pd.Series([x[0] for x in z]).corr(pd.Series([x[1] for x in z])); ic.append(q);obs.append(len(z)); regs.setdefault(dt.year,[]).append(q)
  for k in (5,10):
   zz=[(v,out[s][k]) for s,v in vals.items() if s in out and k in out[s]]
   hs[k].append(pd.Series([x[0] for x in zz]).corr(pd.Series([x[1] for x in zz])) if len(zz)>=8 else np.nan)
  rr=pd.Series(vals).rank(pct=True)
  if prev is not None: turns += [abs(rr[s]-prev[s]) for s in set(rr.index)&set(prev.index)]
  prev=rr

def st(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)];return len(a),float(np.mean(a)),float(np.mean(a)/np.std(a,ddof=1)),float(np.mean(a>0))
print('dates',len(ic),'avg_names',np.mean(obs),'coverage',len(ic)/len(all_dates),'turnover',np.mean(turns))
print('daily',st(ic))
for k,v in hs.items():print(str(k)+'d',st(v))
print('regimes',{y:st(v) for y,v in regs.items()})
print('valid_assets',len(D),'total_dates',len(all_dates))
