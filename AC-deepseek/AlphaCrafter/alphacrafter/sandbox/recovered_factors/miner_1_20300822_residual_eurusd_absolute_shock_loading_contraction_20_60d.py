"""One idea: residual EURUSD absolute-shock loading contraction, 20d versus 60d.
Tests whether assets whose idiosyncratic returns have recently become less sensitive
than usual to large EURUSD moves exhibit cross-sectional forward returns.
"""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0].replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2030-08-21')")
exec(prefix,globals())
eur=pd.read_csv('../persistent/index_data/EURUSD.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change().reindex(p.index)
# Absolute FX shock is a currency-volatility/liquidity impulse; loading uses all days.
x=eur.abs()
def beta(w,n):
 out=pd.DataFrame(np.nan,index=p.index,columns=A)
 for a in A:
  out[a]=e[a].rolling(w,min_periods=n).cov(x)/(x.rolling(w,min_periods=n).var()+1e-12)
 return out
f=beta(60,35)-beta(20,12) # positive means recent sensitivity contracted from structural level
print('FACTOR residual_eurusd_absolute_shock_loading_contraction_20_60d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; pairs=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): pairs.append((t,q)); ns.append(len(z))
 z=pd.Series(dict(pairs)); ics[h]=z; sd=z.std(ddof=1)
 metrics[h]={'daily_paper_ic':z.mean(),'daily_paper_icir':z.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(z)),'ic_hit_ratio':(z>0).mean(),'ic_dates':len(z),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 z=ics[10][mask]; print('REGIME10',name,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
rk=f.rank(axis=1,pct=True); to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman'); screen.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',n,'rho',round(rho,6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']}for h,q in metrics.items()}))
