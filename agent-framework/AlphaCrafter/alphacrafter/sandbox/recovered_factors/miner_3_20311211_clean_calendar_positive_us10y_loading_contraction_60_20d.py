"""One candidate, clean-calendar audit: US10Y positive-change loading contraction (60d vs 20d).
All assets are carried forward onto a common business-day decision calendar before returns,
so a forward horizon is defined consistently and does not require coincident market holidays."""
import json,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-12-10')
def load(a,c): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
cal=pd.bdate_range('2020-01-01',END)
p=pd.DataFrame({a:load(a,'close').reindex(cal).ffill() for a in A})
vol=pd.DataFrame({a:load(a,'volume').reindex(cal).ffill() for a in A})
r=p.pct_change();m=r.mean(axis=1);own=r.rolling(20,min_periods=15).std()
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/(y.rolling(w,min_periods=n).var()+1e-12) for a in A})
def residual(y,*xs):
 out=pd.DataFrame(np.nan,index=y.index,columns=A)
 for t in y.index:
  z=pd.DataFrame({'y':y.loc[t],**{str(i):x.loc[t] for i,x in enumerate(xs)}}).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]: out.loc[t,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return out
b60=beta(r,m,60,40);e=r-b60.mul(m,axis=0);lv=np.log(vol.replace(0,np.nan));vs=lv-lv.rolling(20,min_periods=15).mean()
# Reconstruct prior research library signals on the identical clean calendar.
trend=(p/p.shift(20)-1)/own
lib={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),'miner_1_vol_of_vol_cv20':r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).std()/r.rolling(5,min_periods=4).std().rolling(20,min_periods=15).mean(),'miner_3_relative_volume_participation_20d':np.log(vol/vol.rolling(20,min_periods=15).mean())}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].reindex(cal).ffill().pct_change();lib['miner_1_residualized_vix_stress_resilience_beta20']=residual(-beta(r,vix,20,15),own)
# Existing macro-transmission family, sufficient to detect mechanical loading overlap.
def rollbeta(d,w,n): return pd.DataFrame({a:e[a].rolling(w,min_periods=n).cov(d)/(d.rolling(w,min_periods=n).var()+1e-12) for a in A})
oil=r.WTI.clip(lower=0);lib['miner_3_positive_oil_loading_contraction']=rollbeta(oil,60,42)-rollbeta(oil,20,14)
cu=r.COPPER.clip(lower=0);lib['miner_3_positive_copper_loading_expansion']=rollbeta(cu,20,14)-rollbeta(cu,60,42)
crypto=r[['BTC','ETH']].mean(axis=1);lib['miner_2_crypto_downside_loading_contraction']=rollbeta(crypto.clip(upper=0),60,42)-rollbeta(crypto.clip(upper=0),20,14)
lib['miner_2_crypto_upside_loading_contraction']=rollbeta(crypto.clip(lower=0),60,42)-rollbeta(crypto.clip(lower=0),20,14)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].reindex(cal).ffill().pct_change();lib['miner_2_dxy_loading_transition']=beta(r,dxy,60,30)-beta(r,dxy,20,12)
# Candidate only: positive direct yield moves; recent loading lower than structural loading.
driver=r.US10Y.clip(lower=0);f=rollbeta(driver,60,42)-rollbeta(driver,20,14)
print('FACTOR clean_calendar_residual_positive_us10y_direct_change_loading_contraction_60_20d','validation_end',END.date(),'calendar_dates',len(cal),'universe',len(A),'screen_library',len(lib),'driver_nonzero',round(float((driver>0).mean()),4))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1);out=[];ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):out.append((t,z));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<'2025-01-01'),('2025_26',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(float(x.mean()),6),'ICIR',round(float(x.mean()/x.std(ddof=1)),6),'hit',round(float((x>0).mean()),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
s=[]
for n,x in lib.items():
 q=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman')
 if pd.notna(rho):s.append((abs(rho),n,rho,len(q)))
mx,n,rho,c=max(s);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(x['daily_paper_ic']),6),'icir':round(float(x['daily_paper_icir']),6),'dates':int(x['ic_dates'])}for h,x in metrics.items()}))
