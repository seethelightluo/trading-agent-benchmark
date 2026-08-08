"""miner_2 quarterly revalidation: residual downside serial reversal 60d."""
import pathlib,json,numpy as np,pandas as pd
src=pathlib.Path('scripts/miner_3_20270422_residual_upside_market_down_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2027-04-22')","END=pd.Timestamp('2027-10-20')")
exec(src)
# exact admitted definition: negated lag-one autocorrelation of downside beta-neutral residuals
f=pd.DataFrame(np.nan,index=e.index,columns=A)
for a in A:
 x=e[a].clip(upper=0)
 f[a]=-x.rolling(60,min_periods=45).corr(x.shift(1))
print('\nREVALIDATION miner_2_residual_downside_serial_reversal_60d','cutoff',END.date(),'universe',len(A))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==20:
  for name,mask in [('2020_24',x.index<'2025'),('2025_27',x.index>='2025'),('recent_2027',x.index>='2027-01-01')]:
   y=x[mask];print('REGIME',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.mean(tos),6),'TURNOVER_DATES',len(tos),'LATEST_VALID',int(f.iloc[-1].notna().sum()))
mx=-1;win=None
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',name,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);win=name
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',win,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
