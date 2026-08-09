"""miner_1 one candidate: residualized rate-basket correlation coupling (60/20)."""
import pathlib,json,numpy as np,pandas as pd
base=pathlib.Path('scripts/miner_2_20280127_residual_kurtosis_containment_60d.py').read_text()
loader=base.replace("END=pd.Timestamp('2028-01-26')","END=pd.Timestamp('2028-03-08')")
exec(loader.split('# Higher score denotes')[0])
# Positive score: linkage to the two 10-year-rate series has recently increased,
# net of own trend and volatility. This measures emergence of a rate-sensitive driver.
rates=r[['US10Y','CN10Y']].mean(axis=1)
c20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(rates) for a in A})
c60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(rates) for a in A})
trend=(p/p.shift(20)-1)/own
f=-residual(-(c20-c60),trend,own)
# Full inherited admitted-proxy reconstruction used by the most-recent Miner 1 work.
start=base.index('# Reconstruct remaining admitted signals absent from inherited baseline.')
end=base.index("print('FACTOR",start)
exec(base[start:end])
# Factors admitted after that inherited snapshot.
defensive=r[['XAU','US10Y','CN10Y']].mean(axis=1)
d20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(defensive) for a in A}); d60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(defensive) for a in A})
lib['miner_1_residualized_defensive_correlation_coupling_60_20']=residual(-(d20-d60),trend,own)
infl=r[['COPPER','WTI']].mean(axis=1); i20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(infl) for a in A}); i60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(infl) for a in A})
lib['miner_1_residualized_inflation_basket_correlation_decoupling_60_20']=residual(i20-i60,trend,own)
growth=r[['NDX','SOX']].mean(axis=1); g20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(growth) for a in A}); g60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(growth) for a in A})
lib['miner_1_residualized_equity_growth_basket_correlation_coupling_60_20']=residual(-(g20-g60),trend,own)
print('FACTOR residualized_rate_basket_correlation_coupling_60_20 validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'admitted_library_proxies',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); sd=x.std(ddof=1); q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==10:
  for n,m in [('2020_21',x.index<'2022'),('2022_23',(x.index>='2022')&(x.index<'2024')),('2024_25',(x.index>='2024')&(x.index<'2026')),('2026_28',x.index>='2026')]:
   y=x[m]; print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.nanmean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;win=None;cells=0
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',name,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);win=name;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',win,'CELLS',cells,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
