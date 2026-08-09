"""miner_2: validate cross-asset market synchronization improvement (one factor)."""
import json,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# date supplied by controller; last completed visible row avoids future data
END=pd.Timestamp('2026-12-16')
def get(a,c='close', folder='stock_data'):
 d=pd.read_csv(f'../persistent/{folder}/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,c].astype(float)
p=pd.DataFrame({a:get(a) for a in A}); r=p.pct_change(fill_method=None); rv=r.rolling(20,min_periods=15).std(); vol=pd.DataFrame({a:get(a,'volume') for a in A})
# An asset is favored after its medium-window correlation with the equal-weight cross-asset market has fallen over 20 sessions.
# Unlike downside beta this considers both risk-on and risk-off synchronization.
market=r.mean(axis=1); co=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(market) for a in A}); f=co.shift(20)-co
lib={}
trend=(p/p.shift(20)-1)/rv
lib['risk_adjusted_trend_20d']=trend;lib['ravmom_20obs']=trend
lib['volnorm_reversal_5obs']=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
lib['vol_of_vol_cv20']=-r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean()
lib['relative_volume_participation_20d']=vol/vol.rolling(20,min_periods=15).mean()
vix=get('VIX','close','index_data').pct_change(); vb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(vix)/vix.rolling(20,min_periods=15).var() for a in A});vx=pd.DataFrame(index=p.index,columns=A)
for dt in p.index:
 z=pd.DataFrame({'y':vb.loc[dt],'x':rv.loc[dt]}).dropna()
 if len(z)>=8 and z.x.var()>0:
  X=np.c_[np.ones(len(z)),z.x];vx.loc[dt,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
lib['residualized_vix_stress_resilience_beta20']=vx
down=market.where(market<0); beta=pd.DataFrame({a:r[a].rolling(120,min_periods=30).cov(down)/down.rolling(120,min_periods=30).var() for a in A});lib['downside_beta_improvement_120_20']=beta.shift(20)-beta
breadth=(p/p.rolling(60,min_periods=40).max()<.95).mean(axis=1).astype(float); shock=breadth.diff(); dc=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(shock) for a in A});lib['drawdown_synchronization_improvement_60_20']=dc.shift(20)-dc
print('FACTOR market_synchronization_improvement_60_20 END',END.date(),'UNIVERSE',len(A))
out={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.x.nunique()>1: vals.append((dt,z.x.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};out[str(h)+'d']=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 for n,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  y=x[mask]
  if len(y):print('REGIME',h,n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'TURNOVER',round(float(np.mean(ts)),6),'TURNOVER_DATES',len(ts))
mx=0
for n,s in lib.items():
 z=pd.concat([f.stack().rename('x'),s.stack().rename('y')],axis=1).dropna();rho=z.x.corr(z.y,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6));print('DECAY',json.dumps(out))
