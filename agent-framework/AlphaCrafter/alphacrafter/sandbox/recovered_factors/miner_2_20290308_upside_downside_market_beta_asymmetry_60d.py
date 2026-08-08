"""Validate one pre-specified factor: 60d upside-minus-downside market beta asymmetry.
Uses only bars through 2029-03-07 and reconstructs the admitted library for mandatory screen.
"""
import pathlib, json, numpy as np, pandas as pd
src=pathlib.Path('scripts/miner_3_20290111_residual_defensive_basket_correlation_contraction_60_20d.py').read_text()
# Reuse its audited data/library reconstruction only; omit its candidate and reporting section.
prefix=src.split('# Candidate: recent contraction versus structural correlation')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-01-10')", "END=pd.Timestamp('2029-03-07')")
exec(prefix)
# Idea: market participation asymmetry.  Higher means favorable-upside exposure relative to downside exposure.
R=r.to_numpy(); M=m.to_numpy(); arr=np.full(R.shape,np.nan)
for t in range(len(r)):
    if t < 59: continue
    mm=M[t-59:t+1]
    for k in range(len(A)):
        rr=R[t-59:t+1,k]
        good=np.isfinite(rr)&np.isfinite(mm)&(mm>0)
        bad=np.isfinite(rr)&np.isfinite(mm)&(mm<0)
        if good.sum()>=12 and bad.sum()>=12 and np.var(mm[good])>0 and np.var(mm[bad])>0:
            bu=np.cov(rr[good],mm[good],ddof=1)[0,1]/np.var(mm[good])
            bd=np.cov(rr[bad],mm[bad],ddof=1)[0,1]/np.var(mm[bad])
            arr[t,k]=bu-bd
f=pd.DataFrame(arr,index=p.index,columns=A)
print('FACTOR upside_minus_downside_market_beta_asymmetry_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; obs=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): obs.append((t,q)); ns.append(len(z))
 x=pd.Series(dict(obs));ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<'2025'),('2025_26',(ics[10].index>='2025')&(ics[10].index<'2027')),('2027_29',(ics[10].index>='2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
res=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman');res.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(res);print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',n,'rho',round(rho,6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
