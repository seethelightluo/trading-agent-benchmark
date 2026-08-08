"""Scheduled complete-library revalidation of Residual Broad-Drawdown Resilience (60d), visible through 2028-08-23."""
import pathlib,json,numpy as np,pandas as pd,io,contextlib
src=pathlib.Path('scripts/miner_3_20280504_residual_downside_volume_deceleration_complete_library.py').read_text().replace("END=pd.Timestamp('2028-05-03')","END=pd.Timestamp('2028-08-23')")
with contextlib.redirect_stdout(io.StringIO()): exec(src)
# 60-session residual performance conditional on weak cross-asset breadth.
market=r.mean(axis=1)
beta=r.rolling(60,min_periods=40).cov(market).div(market.rolling(60,min_periods=40).var(),axis=0)
e=r-beta.mul(market,axis=0)
breadth=(r>0).mean(axis=1)
stress=(breadth<=.40).astype(float)
num=e.mul(stress,axis=0).rolling(60,min_periods=40).sum()
den=stress.rolling(60,min_periods=40).sum().replace(0,np.nan)
f=num.div(den,axis=0)
key='miner_2_residual_broad_drawdown_resilience_60d'; lib[key]=f
print('REVALIDATION',key,'END',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'other_library',len(lib)-1)
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p)-1; vals=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): vals.append((t,q));ns.append(len(z))
 x=pd.Series(dict(vals),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
x=ics[20]
for name,mask in [('2025_2026',(x.index>=pd.Timestamp('2025-01-01'))&(x.index<pd.Timestamp('2027-01-01'))),('2027_2028_08_23',x.index>=pd.Timestamp('2027-01-01'))]:
 y=x[mask];print('REGIME20',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),6))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(tos)),6),'TURNOVER_DATES',len(tos),'LATEST_VALID',int(f.iloc[-1].notna().sum()),'STRESS_FREQUENCY',round(float(stress.mean()),6))
mx=-1;winner=None;cells=0
for n,s in sorted(lib.items()):
 if n==key:continue
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman')
 if abs(rho)>mx:mx=abs(rho);winner=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',cells)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'hit':round(q['ic_hit_ratio'],6),'dates':q['ic_dates'],'names':round(q['mean_valid_instruments'],4)} for h,q in metrics.items()}))
