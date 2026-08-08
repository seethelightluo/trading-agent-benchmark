"""Revalidate one candidate: inverse residual VIX-EURUSD joint-stress loading expansion with complete 30-factor library screen."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0].replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2031-02-19')")
exec(prefix,globals())
# Add all three post-2029 admitted signals absent from inherited 27-signal audit library.
def z60(x): return (x-x.rolling(60,min_periods=40).mean())/(x.rolling(60,min_periods=40).std()+1e-12)
# commodity-currency divergence expansion
D=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].reindex(p.index).ffill().pct_change()
imp=((r['COPPER']-D)/(r['COPPER']-D).rolling(60,min_periods=40).std()+1e-12)).clip(lower=0)
lib['miner_2_residual_commodity_currency_divergence_loading_expansion_20_60d']=beta(e,imp,20,14)-beta(e,imp,60,42)
# commodity-crypto dispersion compression loading expansion
cmd=r[['XAU','COPPER','WTI']].mean(axis=1); cry=r[['BTC','ETH']].mean(axis=1); comp=(-z60(cmd-cry).abs())
lib['miner_2_residual_commodity_crypto_dispersion_compression_loading_expansion_20_60d']=beta(e,comp,20,14)-beta(e,comp,60,42)
# defensive-cyclical compression loading contraction
defe=r[['XAU','US10Y','CN10Y']].mean(axis=1); cyc=r[['SPX','SX5E','SOX','NDX','COPPER','WTI']].mean(axis=1); dc=-z60(defe-cyc).abs()
lib['miner_2_residual_defensive_cyclical_dispersion_compression_loading_contraction_20_60d']=-(beta(e,dc,20,14)-beta(e,dc,60,42))
# Candidate: high means reduced recent loading relative to structural loading in VIX-up/EURUSD-down episodes.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].reindex(p.index).ffill().pct_change()
eur=pd.read_csv('../persistent/index_data/EURUSD.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].reindex(p.index).ffill().pct_change()
driver=(z60(vix).clip(lower=0,upper=5)*(-z60(eur)).clip(lower=0,upper=5)).fillna(0.)
f=beta(e,driver,60,42)-beta(e,driver,20,14)
print('FACTOR inverse_residual_vix_eurusd_joint_stress_loading_expansion_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_screened',len(lib),'driver_nonzero_fraction',round(float((driver!=0).mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   ic=q.f.corr(q.y,method='spearman')
   if pd.notna(ic):out.append((t,ic));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2025_26',(ics[20].index>=pd.Timestamp('2025'))&(ics[20].index<pd.Timestamp('2027'))),('2027_onward',ics[20].index>=pd.Timestamp('2027'))]:
 x=ics[20][mask];print('REGIME20',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(ts)),6),'TURNOVER_DATES',len(ts))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=q.f.corr(q.s,method='spearman');screen.append((abs(rho),n,rho,len(q)))
mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']} for h,v in metrics.items()}))
