"""One candidate: positive USDJPY-shock residual loading contraction (20d vs 60d)."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0].replace("END=pd.Timestamp('2029-02-21')","END=pd.Timestamp('2031-01-22')")
exec(prefix,globals())
# Complete the reconstructed library with factors admitted after the inherited snapshot.
def rollbeta(driver,w,n): return pd.DataFrame({a:e[a].rolling(w,min_periods=n).cov(driver)/(driver.rolling(w,min_periods=n).var()+1e-12) for a in A})
# Commodity/crypto compression expansion
oil=r['WTI'].clip(lower=0); lib['residual_positive_oil_change_shock_loading_contraction_20_60d']=rollbeta(oil,60,42)-rollbeta(oil,20,14)
relcc=r[['XAU','COPPER','WTI']].mean(axis=1)-r[['BTC','ETH']].mean(axis=1); zcc=(relcc-relcc.rolling(60,min_periods=40).mean())/(relcc.rolling(60,min_periods=40).std()+1e-12); drv=(-zcc.abs()).clip(-5,0).fillna(0)
lib['miner_2_residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d']=rollbeta(drv,20,14)-rollbeta(drv,60,42)
# residual positive copper shock expansion
cu=r['COPPER'].clip(lower=0);lib['residual_positive_copper_change_shock_loading_expansion_20_60d']=rollbeta(cu,20,14)-rollbeta(cu,60,42)
# defensive/cyclical compression contraction
sp=r[['XAU','US10Y','CN10Y']].mean(axis=1)-r[['SPX','SX5E','SOX','NDX','COPPER','WTI']].mean(axis=1); zz=(sp-sp.rolling(60,min_periods=40).mean())/(sp.rolling(60,min_periods=40).std()+1e-12); drv2=(-zz.abs()).clip(-5,0).fillna(0)
lib['miner_2_residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d']=rollbeta(drv2,60,42)-rollbeta(drv2,20,14)
# crypto downside transmission contraction
cd=r[['BTC','ETH']].mean(axis=1).clip(upper=0);lib['miner_2_residual_crypto_downside_transmission_contraction_60_20d']=rollbeta(cd,60,42)-rollbeta(cd,20,14)
# Library-orthogonal crypto upside transmission (the candidate signal was constructed from residual returns).
cupr=r[['BTC','ETH']].mean(axis=1).clip(lower=0); lib['miner_2_library_orthogonal_residual_crypto_upside_transmission_contraction_60_20d']=rollbeta(cupr,60,42)-rollbeta(cupr,20,14)
# downside-state defensive basket transition
down=(m<0).astype(float); defensive=e[['XAU','US10Y','CN10Y']].mean(axis=1)*down
lib['miner_2_downside_state_residual_defensive_basket_beta_transition_60_20d']=rollbeta(defensive,60,42)-rollbeta(defensive,20,14)
# Candidate: sensitivity to directly observable *positive* yen weakening shocks,
# reduced recently relative to structural sensitivity. This differs from the prior
# residualized yen-innovation expansion and avoids its sparse regression warmup.
fx=pd.read_csv('../persistent/index_data/USDJPY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].reindex(p.index).ffill().pct_change()
shock=fx.clip(lower=0)
f=rollbeta(shock,60,42)-rollbeta(shock,20,14)
print('FACTOR positive_usdjpy_change_shock_loading_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'shock_nonzero_fraction',round(float((shock>0).mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v):out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman');screen.append((abs(rho),n,rho,len(q)))
mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))
