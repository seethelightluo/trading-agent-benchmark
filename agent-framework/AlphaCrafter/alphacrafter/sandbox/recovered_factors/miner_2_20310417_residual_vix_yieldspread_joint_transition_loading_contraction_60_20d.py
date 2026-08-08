"""One candidate: residual VIX--US/CN yield-spread transition loading contraction (60d vs 20d)."""
import json, numpy as np, pandas as pd
# Reuse the visibility-safe data setup and reconstructed admitted-library signals.
src=open('scripts/miner_2_20310403_residual_dxy_eurusd_divergence_loading_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Continuous, signed dollar-versus-euro')[0].replace("END=pd.Timestamp('2031-04-02')","END=pd.Timestamp('2031-04-16')")
exec(prefix,globals())
# A continuous macro-transition driver: standardized VIX change times standardized
# change in the observed US-minus-China 10y yield spread. This represents a joint
# volatility/rates transition, rather than equity market direction or FX shock.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change().reindex(p.index)
us=pd.read_csv('../persistent/stock_data/US10Y.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).reindex(p.index)
cn=pd.read_csv('../persistent/stock_data/CN10Y.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).reindex(p.index)
spread=(us-cn).diff()
zv=vix/(vix.rolling(60,min_periods=40).std()+1e-12)
zs=spread/(spread.rolling(60,min_periods=40).std()+1e-12)
driver=zv*zs
# Higher score: recent idiosyncratic exposure to this joint transition is lower
# than its structural exposure. Asset returns are first residualized to broad market.
f=beta(e,driver,60,42)-beta(e,driver,20,14)
print('FACTOR residual_vix_yieldspread_joint_transition_loading_contraction_60_20d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_reconstructed',len(lib),'driver_nonnull',round(driver.notna().mean(),6))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q)); ns.append(len(z))
 z=pd.Series(dict(out)); ics[h]=z; sd=z.std(ddof=1)
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
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',n,'rho',round(rho,6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
