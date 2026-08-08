"""Miner 1: residualized FX-linkage acceleration (60/20), cutoff 2028-09-06."""
import pathlib,json,numpy as np,pandas as pd,io,contextlib
# Point-in-time safe full-library panel; underlying loader is capped before this decision date.
src=pathlib.Path('scripts/miner_3_20280504_residual_downside_volume_deceleration_complete_library.py').read_text().replace("END=pd.Timestamp('2028-05-03')","END=pd.Timestamp('2028-09-06')")
with contextlib.redirect_stdout(io.StringIO()): exec(src)
# A signed broad-FX factor: dollar strength versus CNY and broad EUR/JPY weakness.
def fxret(name):
 q=pd.read_csv('../persistent/index_data/'+name+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change()
 return q.reindex(r.index)
usdcny=fxret('USDCNY'); usdjpy=fxret('USDJPY'); eurusd=fxret('EURUSD')
fx=pd.concat([usdcny,usdjpy,-eurusd],axis=1).mean(axis=1)
# Higher score: a recently increasing loading on the FX stress factor, net of trend/risk.
b20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(fx)/fx.rolling(20,min_periods=15).var() for a in A})
b60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(fx)/fx.rolling(60,min_periods=40).var() for a in A})
trend=(p/p.shift(20)-1)/own
f=residual(b20-b60,trend,own)
# Add effective signals admitted after the May complete-library snapshot.
lv=np.log(vol.replace(0,np.nan)); vs=lv-lv.rolling(20,min_periods=15).mean()
svp=(np.sign(e)*lv.diff()).where(e<0)
lib['miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d']=-(svp.rolling(20,min_periods=8).mean()-svp.rolling(60,min_periods=18).mean())/(e.rolling(60,min_periods=40).std()+1e-12)
particip=vol.div(vol.rolling(20,min_periods=15).mean())
dd=(p/p.rolling(60,min_periods=40).max()-1).abs()
lib['miner_3_drawdown_weighted_relative_participation_rank_acceleration_20_60d']=(particip.rolling(20,min_periods=12).mean()-particip.rolling(60,min_periods=30).mean())*dd
# Candidate must be compared to every effective signal, removing obsolete/deprecated proxies only.
effective={
'miner_2_drawdown_synchronization_improvement_60_20','miner_1_residualized_return_autocorrelation_20d','miner_1_residualized_vix_stress_resilience_beta20','market_synchronization_increase_60_20','miner_3_relative_volume_participation_20d','miner_3_realized_volatility_compression_20_60d','miner_1_residualized_defensive_correlation_decoupling_60_20','miner_2_residual_upside_serial_reversal_60d','miner_3_risk_adjusted_trend_20d','miner_1_residualized_inflation_basket_correlation_decoupling_60_20','miner_1_residualized_equity_growth_basket_correlation_coupling_60_20','miner_3_residual_downside_volume_confirmation_deceleration_20_60d','miner_1_vol_of_vol_cv20','miner_3_residual_median_minus_mean_60d','miner_1_market_beta_contraction_60_20','miner_1_residualized_realized_return_skewness_20d','miner_3_residual_upside_volume_confirmation_deceleration_20_60d','miner_1_ravmom_20obs','miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d','residualized_downside_tail_containment_20','miner_3_residual_dispersion_shock_resilience_60d','miner_3_residual_lower_partial_moment_60d','miner_1_breadth_recovery_capture_60d','miner_3_drawdown_weighted_relative_participation_rank_acceleration_20_60d','miner_3_residual_upside_volume_confirmation_60d','miner_1_volnorm_reversal_5obs','miner_1_residualized_drawdown_recovery_60_10'}
# base reconstruction names are identifiers, so retain only verified present effective entries.
lib={k:v for k,v in lib.items() if k in effective}
print('FACTOR residualized_fx_stress_linkage_acceleration_60_20','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'effective_library_proxies',len(lib))
metrics={}; IC={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p)-1; out=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q):out.append((t,q));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);IC[h]=x;sd=x.std(ddof=1)
 q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
x=IC[10]
for n,m in [('2020_21',x.index<'2022'),('2022_23',(x.index>='2022')&(x.index<'2024')),('2024_25',(x.index>='2024')&(x.index<'2026')),('2026_28',x.index>='2026')]:
 y=x[m];print('REGIME10',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
mx=-1;winner=None;cells=0
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman')
 print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
 if len(z) and abs(rho)>mx:mx=abs(rho);winner=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',cells)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'hit':round(q['ic_hit_ratio'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
