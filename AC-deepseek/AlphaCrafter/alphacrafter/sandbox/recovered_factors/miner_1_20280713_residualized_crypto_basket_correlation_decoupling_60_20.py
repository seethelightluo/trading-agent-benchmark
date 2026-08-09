"""miner_1: residualized crypto-basket correlation decoupling (60/20)."""
import pathlib,json,numpy as np,pandas as pd
# PIT loader: data are explicitly limited to the last completed session.
src=pathlib.Path('scripts/miner_1_20280615_residualized_idiosyncratic_capture_asymmetry_60.py').read_text()
src=src.replace("END=pd.Timestamp('2028-06-14')","END=pd.Timestamp('2028-07-12')")
exec(src.split("# A high score indicates")[0])
# High score denotes a fall in 20-session co-movement with the BTC/ETH basket
# relative to 60 sessions. Trend and own realized risk are removed cross-sectionally,
# isolating changing crypto-linkage rather than simple momentum or low volatility.
basket=r[['BTC','ETH']].mean(axis=1)
c20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(basket) for a in A})
c60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(basket) for a in A})
trend=(p/p.shift(20)-1)/own
f=residual(-(c20-c60),trend,own)
# Reconstruct every currently admitted signal other than this candidate.
lv=np.log(vol.replace(0,np.nan));vs=lv-lv.rolling(20,min_periods=15).mean()
lib['miner_2_downside_vs_upside_volume_change_60d']=(lv.diff().where(r<0).rolling(60,min_periods=12).mean()-lv.diff().where(r>0).rolling(60,min_periods=12).mean())
down=e.clip(upper=0);lib['miner_2_residual_downside_serial_reversal_60d']=pd.DataFrame({a:-down[a].rolling(60,min_periods=45).corr(down[a].shift(1)) for a in A})
B=(r>0).mean(axis=1);shock=B.diff().clip(lower=0);lib['miner_1_breadth_recovery_capture_60d']=pd.DataFrame({a:e[a].rolling(60,min_periods=40).cov(shock)/shock.rolling(60,min_periods=40).var() for a in A})
lib['miner_3_realized_volatility_compression_20_60d']=-(r.rolling(20,min_periods=15).std()/(r.rolling(60,min_periods=40).std()+1e-12))
lib['miner_1_residualized_realized_return_skewness_20d']=pd.DataFrame({a:e[a].rolling(20,min_periods=15).skew() for a in A})
disp=r.std(axis=1,ddof=0).diff();lib['miner_3_residual_dispersion_shock_resilience_60d']=pd.DataFrame({a:-e[a].rolling(60,min_periods=45).corr(disp) for a in A})
lib['miner_3_residual_upside_volume_confirmation_60d']=(e.clip(lower=0)*vs.clip(lower=0)).rolling(60,min_periods=18).mean()/(e.rolling(60,min_periods=40).std()+1e-12)
breadth=(r>0).mean(axis=1);stress=breadth<=.40
lib['miner_2_residual_broad_drawdown_resilience_60d']=-e.where(stress).rolling(60,min_periods=12).std()/(e.rolling(60,min_periods=40).std()+1e-12)
lib['miner_3_residual_upside_volume_confirmation_acceleration_20_60d']=((e.clip(lower=0)*vs.clip(lower=0)).rolling(20,min_periods=8).mean()-(e.clip(lower=0)*vs.clip(lower=0)).rolling(60,min_periods=18).mean())/(e.rolling(60,min_periods=40).std()+1e-12)
lib['miner_3_residual_downside_volume_confirmation_deceleration_20_60d']=-(((-e).clip(lower=0)*vs.clip(lower=0)).rolling(20,min_periods=8).mean()-((-e).clip(lower=0)*vs.clip(lower=0)).rolling(60,min_periods=18).mean())/(e.rolling(60,min_periods=40).std()+1e-12)
# June-2028 admitted downside signed-volume-pressure deceleration.
svp=(np.sign(e)*lv.diff()).where(e<0)
lib['miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d']=-(svp.rolling(20,min_periods=8).mean()-svp.rolling(60,min_periods=18).mean())/(e.rolling(60,min_periods=40).std()+1e-12)
print('FACTOR residualized_crypto_basket_correlation_decoupling_60_20 validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'other_admitted_library',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;vals=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==10:
  for n,m in [('2020_21',x.index<'2022'),('2022_23',(x.index>='2022')&(x.index<'2024')),('2024_25',(x.index>='2024')&(x.index<'2026')),('2026_28',x.index>='2026')]:
   y=x[m];print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.nanmean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;win=None;cells=0
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);win=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',win,'CELLS',cells,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
