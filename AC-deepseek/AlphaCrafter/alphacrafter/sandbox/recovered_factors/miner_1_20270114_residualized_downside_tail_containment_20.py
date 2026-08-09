"""One-idea test: residualized downside-tail containment (20d), visible through 2027-01-13."""
import numpy as np,pandas as pd,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-01-13')
def col(a,c='close'):
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,c].astype(float)
p=pd.DataFrame({a:col(a) for a in A}); r=p.pct_change(fill_method=None); volume=pd.DataFrame({a:col(a,'volume') for a in A})
rv=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/rv
# Candidate: negative standardized mean loss on down sessions; CS residual versus trend and total volatility.
loss=(-r.where(r<0)).rolling(20,min_periods=6).mean()/rv
f=pd.DataFrame(np.nan,index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':-loss.loc[dt],'trend':trend.loc[dt],'vol':rv.loc[dt]}).dropna()
 if len(z)>=8 and z[['trend','vol']].nunique().min()>1:
  X=np.c_[np.ones(len(z)),z.trend,z.vol];f.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
# exact/reasonable reconstructions of all 10 admitted signals
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':volume/volume.rolling(20,min_periods=15).mean()}
vix=col('../persistent/index_data/VIX'.replace('../persistent/stock_data/',''),'close') if False else pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change()
beta=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A});vx=pd.DataFrame(np.nan,index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':beta.loc[dt],'v':rv.loc[dt]}).dropna()
 if len(z)>=8 and z.v.nunique()>1:
  X=np.c_[np.ones(len(z)),z.v];vx.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['miner_1_residualized_vix_stress_resilience_beta20']=vx
m=r.mean(axis=1); down=m.where(m<0);db=pd.DataFrame({a:r[a].rolling(120,min_periods=30).cov(down)/down.rolling(120,min_periods=30).var() for a in A});lib['miner_2_downside_beta_improvement_120_20']=db.shift(20)-db
breadth=(p/p.rolling(60,min_periods=45).max()<.95).mean(axis=1); ds=breadth.diff(); sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(ds) for a in A});lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy
mk=r.mean(axis=1); co=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(mk) for a in A});lib['miner_2_market_synchronization_increase_60_20']=co-co.shift(20)
# recovery definition as persisted
DD=p/p.rolling(60,min_periods=45).max()-1;raw=(p/p.shift(10)-1)*(-DD.clip(upper=0)); rec=pd.DataFrame(np.nan,index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':raw.loc[dt],'trend':trend.loc[dt],'vol':rv.loc[dt]}).dropna()
 if len(z)>=8 and z[['trend','vol']].nunique().min()>1:
  X=np.c_[np.ones(len(z)),z.trend,z.vol];rec.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['miner_1_residualized_drawdown_recovery_60_10']=rec
print('FACTOR residualized_downside_tail_containment_20 END',END.date(),'universe',len(A),'panel',p.index.min().date(),p.index.max().date())
res={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;obs=[];nn=[]
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.x.nunique()>1:obs.append((dt,z.x.corr(z.y,method='spearman')));nn.append(len(z))
 x=pd.Series(dict(obs));sd=x.std(ddof=1);res[h]=x
 print('H',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round((x>0).mean(),4),'dates',len(x),'mean_n',round(np.mean(nn),2),'se',round(sd/np.sqrt(len(x)),6))
x=res[5]
for n,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_27',x.index>='2025-01-01')]:
 y=x[mask];print('REGIME5',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURNOVER',round(np.mean(turn),6),'TURNOVER_DATES',len(turn))
mx=0;ok=True
for n,s in lib.items():
 z=pd.concat([f.stack().rename('x'),s.stack().rename('y')],axis=1).dropna();rho=z.x.corr(z.y,method='spearman') if len(z) else np.nan;ok &=np.isfinite(rho);mx=max(mx,abs(rho)) if np.isfinite(rho) else np.nan;print('LIB',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'COMPLETE',ok)
