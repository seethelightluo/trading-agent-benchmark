"""miner_2: residual broad-drawdown outcome dispersion, 60d; one candidate."""
import pathlib
src=pathlib.Path('scripts/miner_3_20280323_residual_upside_volume_confirmation_acceleration_20_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2028-03-22')","END=pd.Timestamp('2028-04-19')")
# Load only panel/residual/full inherited library construction (prior to candidate).
exec(src.split("# High score:")[0])
# On broad-stress sessions (<=40% assets up), measure dispersion of each asset's
# idiosyncratic outcomes.  Negative orientation rewards stable residual behavior
# under common drawdown rather than the conditional residual mean already admitted.
breadth=(r>0).mean(axis=1); stress=(breadth<=.40)
es=e.where(stress, float('nan'))
f=-es.rolling(60,min_periods=20).std()/(e.rolling(60,min_periods=40).std()+1e-12)
# The inherited source already constructs 23 then its lower block adds 7 current signals.
# Replicate those added signals for full active-library correlation evidence.
lv=np.log(vol.replace(0,np.nan)); vs=lv-lv.rolling(20,min_periods=15).mean()
lib['miner_2_downside_vs_upside_volume_change_60d']=(lv.diff().where(r<0).rolling(60,min_periods=12).mean()-lv.diff().where(r>0).rolling(60,min_periods=12).mean())
down_e=e.clip(upper=0);lib['miner_2_residual_downside_serial_reversal_60d']=pd.DataFrame({a:-down_e[a].rolling(60,min_periods=45).corr(down_e[a].shift(1)) for a in A})
B=(r>0).mean(axis=1);shock=B.diff().clip(lower=0);lib['miner_1_breadth_recovery_capture_60d']=pd.DataFrame({a:e[a].rolling(60,min_periods=40).cov(shock)/shock.rolling(60,min_periods=40).var() for a in A})
lib['miner_3_realized_volatility_compression_20_60d']=-(r.rolling(20,min_periods=15).std()/(r.rolling(60,min_periods=40).std()+1e-12))
lib['miner_1_residualized_realized_return_skewness_20d']=pd.DataFrame({a:e[a].rolling(20,min_periods=15).skew() for a in A})
disp=r.std(axis=1,ddof=0).diff();lib['miner_3_residual_dispersion_shock_resilience_60d']=pd.DataFrame({a:-e[a].rolling(60,min_periods=45).corr(disp) for a in A})
uv=e.clip(lower=0)*vs.clip(lower=0);lib['miner_3_residual_upside_volume_confirmation_60d']=uv.rolling(60,min_periods=18).mean()/(e.rolling(60,min_periods=40).std()+1e-12)
print('FACTOR residual_broad_drawdown_outcome_dispersion_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'admitted_library',len(lib),'stress_frequency',round(float(stress.mean()),6))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:out.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q;print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==10:
  for n,m in [('2020',(x.index<'2021')),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_26',(x.index>='2025')&(x.index<'2027')),('2027_28',x.index>='2027')]:
   y=x[m];print('REGIME',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.nanmean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;win=None
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);win=n
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',win,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))
