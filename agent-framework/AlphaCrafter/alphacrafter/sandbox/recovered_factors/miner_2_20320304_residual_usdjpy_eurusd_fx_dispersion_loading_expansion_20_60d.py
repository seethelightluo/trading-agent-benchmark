"""One candidate: residual USDJPY--EURUSD FX-dispersion loading expansion (20d/60d).
Tests whether rising recent sensitivity of idiosyncratic asset returns to a clean,
observation-only G10 FX dislocation forecasts cross-asset forward returns."""
import json,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-03-03')
def series(path,sym,col):
 return pd.read_csv(path+'/'+sym+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,col].astype(float)
p=pd.DataFrame({a:series('../persistent/stock_data',a,'close') for a in A}).sort_index()
vol=pd.DataFrame({a:series('../persistent/stock_data',a,'volume') for a in A}).reindex(p.index)
r=p.pct_change(); m=r.mean(axis=1)
# Daily cross-sectional residuals remove the contemporaneous common market component.
b60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(m)/(m.rolling(60,min_periods=40).var()+1e-12) for a in A})
e=r-b60.mul(m,axis=0)
def macro(sym):
 x=series('../persistent/index_data',sym,'close').reindex(p.index).ffill()
 return x.pct_change()
def beta_resid(driver,w,n):
 return pd.DataFrame({a:e[a].rolling(w,min_periods=n).cov(driver)/(driver.rolling(w,min_periods=n).var()+1e-12) for a in A})
# A signed G10 divergence innovation: yen weakening versus euro strengthening, standardized on 60 days.
jpy=macro('USDJPY'); eur=macro('EURUSD')
def z(x): return (x-x.rolling(60,min_periods=40).mean())/(x.rolling(60,min_periods=40).std()+1e-12)
driver=(z(jpy)-z(eur)).clip(-6,6)
f=beta_resid(driver,20,14)-beta_resid(driver,60,42)
# Correlation library: rebuild broad, distinct admitted signal families from observable data.
own=r.rolling(20,min_periods=15).std(); lv=np.log(vol.replace(0,np.nan));vs=lv-lv.rolling(20,min_periods=15).mean()
def bret(d,w,n): return pd.DataFrame({a:r[a].rolling(w,min_periods=n).cov(d)/(d.rolling(w,min_periods=n).var()+1e-12) for a in A})
trend=(p/p.shift(20)-1)/(own+1e-12); dd=p/p.rolling(60,min_periods=40).max()-1
vix=macro('VIX'); dxy=macro('DXY'); oil=r.WTI.clip(lower=0); copper=r.COPPER.clip(lower=0)
lib={'trend':trend,'reversal5':-(p/p.shift(5)-1)/(r.rolling(5,min_periods=4).std()+1e-12),'volcompress':-own/(r.rolling(60,min_periods=40).std()+1e-12),'dxybeta':bret(dxy,60,42)-bret(dxy,20,14),'vixbeta':-bret(vix,20,14),'oilbeta':bret(oil,60,42)-bret(oil,20,14),'copperbeta':bret(copper,20,14)-bret(copper,60,42),'skew':e.rolling(20,min_periods=15).skew(),'lowertail':-e.clip(upper=0).rolling(60,min_periods=40).mean()/(e.rolling(60,min_periods=40).std()+1e-12),'relvol':vs,'drawdown':dd,'volume_downup':lv.diff().where(r<0).rolling(60,min_periods=12).mean()-lv.diff().where(r>0).rolling(60,min_periods=12).mean()}
print('FACTOR residual_usdjpy_eurusd_fx_dispersion_loading_expansion_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_proxy',len(lib),'driver_coverage',round(driver.notna().mean(),5),'recent_unique_minmax',int(f.tail(250).nunique(axis=1).min()),int(f.tail(250).nunique(axis=1).max()))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v):out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None,'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.nanmean(turn),6),'TURNOVER_DATES',len(turn),'VALID_CELLS',int(f.notna().sum().sum()))
screen=[]
for n,s in lib.items():
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(q)))
mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION_PROXY',round(mx,6),'FACTOR',n,'rho',round(rho,6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))
