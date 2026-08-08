"""One idea validation: residual downside common-shock co-skewness contraction conditional on cross-asset dispersion stress."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')","END=pd.Timestamp('2029-05-02')")
exec(prefix,globals())
# Add subsequently admitted signals so the independence screen is against the complete active library.
vix_level=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].reindex(p.index).ffill()
market=e.mean(axis=1)
def cosk(a,w,n,shock):
 return (e[a]*shock).rolling(w,min_periods=n).mean()/(e[a].rolling(w,min_periods=n).std()*shock.rolling(w,min_periods=n).std()+1e-12)
vixstress=vix_level>vix_level.rolling(60,min_periods=40).median()
vixdown=(market.where((market<0)&vixstress,0.)**2)
lib['miner_3_vix_stress_residual_downside_coskewness_contraction_20_60d']=pd.DataFrame({a:-(cosk(a,20,14,vixdown)-cosk(a,60,42,vixdown)) for a in A})
rv=r.std(axis=1,ddof=0); rvstress=rv>rv.rolling(60,min_periods=40).median()
rvup=(market.where((market>0)&rvstress,0.)**2)
lib['miner_3_realizedvol_stress_residual_upside_coskewness_contraction_20_60d']=pd.DataFrame({a:-(cosk(a,20,14,rvup)-cosk(a,60,42,rvup)) for a in A})
# Candidate: when cross-asset volatility is elevated, favor assets whose residual
# loading on squared broad downside shocks has recently contracted versus 60d.
# Unlike VIX conditioning it responds to realized cross-asset turbulence.
rvdown=(market.where((market<0)&rvstress,0.)**2)
f=pd.DataFrame({a:-(cosk(a,20,14,rvdown)-cosk(a,60,42,rvdown)) for a in A})
print('FACTOR realizedvol_stress_residual_downside_coskewness_contraction_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'stress_fraction',round(float(rvstress.mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): vals.append((t,q));ns.append(len(z))
 x=pd.Series(dict(vals));ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_2024',ics[20].index<'2025'),('2025_2026',(ics[20].index>='2025')&(ics[20].index<'2027')),('2027_2029',ics[20].index>='2027')]:
 x=ics[20][mask];print('REGIME20',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn),'LATEST_VALID_INSTRUMENTS',int(f.iloc[-1].notna().sum()))
screen=[]
for name,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),name,rho,len(z)))
mx,name,rho,cells=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,'rho',round(float(rho),6),'cells',cells)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))
