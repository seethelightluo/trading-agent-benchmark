"""miner_1: residualized downside-versus-upside defensive linkage asymmetry (60d)."""
import pathlib,json,numpy as np,pandas as pd
rawsrc=pathlib.Path('scripts/miner_1_20280713_residualized_crypto_basket_correlation_decoupling_60_20.py').read_text().replace("END=pd.Timestamp('2028-07-12')","END=pd.Timestamp('2028-07-26')")
exec(rawsrc.split("# High score denotes")[0])
# Within a 60-session window, compare correlation to the internal XAU/rates
# defensive basket in broad-down versus broad-up days. Eight observations per
# conditional subset is the minimum used; trend and own risk are removed.
basket=r[['XAU','US10Y','CN10Y']].mean(axis=1); market_breadth=(r>0).mean(axis=1)
downmask=market_breadth<=.40; upmask=market_breadth>=.60
cd=pd.DataFrame({a:r[a].where(downmask).rolling(60,min_periods=8).corr(basket.where(downmask)) for a in A})
cu=pd.DataFrame({a:r[a].where(upmask).rolling(60,min_periods=8).corr(basket.where(upmask)) for a in A})
trend=(p/p.shift(20)-1)/own
f=residual(-(cd-cu),trend,own)
# Full current library, reconstructed in a separate namespace to avoid mutation.
ns={}; libsrc=rawsrc.split("# Reconstruct every currently admitted signal other than this candidate.")[1]
# It expects f and base variables, while its metric tail is not needed here.
exec(libsrc.split("print('FACTOR")[0],globals(),ns)
lib=ns['lib']
print('FACTOR residualized_downside_vs_upside_defensive_linkage_asymmetry_60 validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'other_admitted_library',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; nsamp=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.y,method='spearman')));nsamp.append(len(z))
 x=pd.Series(dict(vals),dtype=float); sd=x.std(ddof=1); q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(nsamp)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==10:
  for n,m in [('2020_21',x.index<'2022'),('2022_23',(x.index>='2022')&(x.index<'2024')),('2024_25',(x.index>='2024')&(x.index<'2026')),('2026_28',x.index>='2026')]:
   y=x[m]; print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.nanmean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;win=None;cells=0
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);win=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',win,'CELLS',cells,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
