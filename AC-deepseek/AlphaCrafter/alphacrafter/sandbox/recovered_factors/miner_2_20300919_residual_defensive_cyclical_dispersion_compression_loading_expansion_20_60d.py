"""One candidate: residual defensive-cyclical relative-dispersion compression loading expansion (20 vs 60)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2030-09-18')")
exec(prefix,globals())
# Add admitted signals created after the inherited reconstruction, so the screen covers active library.
oil=r['WTI'].clip(lower=0)
lib['residual_positive_oil_change_shock_loading_contraction_20_60d']=pd.DataFrame({a:e[a].rolling(60,min_periods=42).cov(oil)/(oil.rolling(60,min_periods=42).var()+1e-12)-e[a].rolling(20,min_periods=14).cov(oil)/(oil.rolling(20,min_periods=14).var()+1e-12) for a in A})
commod=['XAU','COPPER','WTI']; crypto=['BTC','ETH']; relcc=r[commod].mean(axis=1)-r[crypto].mean(axis=1)
zcc=(relcc-relcc.rolling(60,min_periods=40).mean())/(relcc.rolling(60,min_periods=40).std()+1e-12); compcc=(-zcc.abs()).clip(-5,0).fillna(0.)
lib['miner_2_residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d']=pd.DataFrame({a:e[a].rolling(20,min_periods=14).cov(compcc)/(compcc.rolling(20,min_periods=14).var()+1e-12)-e[a].rolling(60,min_periods=42).cov(compcc)/(compcc.rolling(60,min_periods=42).var()+1e-12) for a in A})
# A cross-regime compression driver: defensive return basket unusually close to cyclical equity/commodity basket.
defensive=['XAU','US10Y','CN10Y']; cyclical=['SPX','SX5E','SOX','NDX','COPPER','WTI']
spread=r[defensive].mean(axis=1)-r[cyclical].mean(axis=1)
z=(spread-spread.rolling(60,min_periods=40).mean())/(spread.rolling(60,min_periods=40).std()+1e-12)
driver=(-z.abs()).clip(-5,0).fillna(0.)
f=pd.DataFrame({a:e[a].rolling(20,min_periods=14).cov(driver)/(driver.rolling(20,min_periods=14).var()+1e-12)-e[a].rolling(60,min_periods=42).cov(driver)/(driver.rolling(60,min_periods=42).var()+1e-12) for a in A})
print('FACTOR residual_defensive_cyclical_dispersion_compression_loading_expansion_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'driver_nonzero_fraction',round(float((driver!=0).mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v):out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(ts)),6),'TURNOVER_DATES',len(ts))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=q.f.corr(q.s,method='spearman');screen.append((abs(rho),n,rho,len(q)))
mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))
