"""miner_2: volume-amplified short reversal residualized to admitted library; one factor idea."""
import json, glob
import numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2026-10-07'; H=[1,5,10,20]
libs={k:{} for k in ['risk_adjusted_trend_20d','relative_volume_participation_20d','ravmom_20obs','volnorm_reversal_5obs','vol_of_vol_cv20','vix_stress_resilience_beta20']}; raw={}; fw={}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
vc=(vix['close'] if 'close' in vix else vix.select_dtypes('number').iloc[:,0]); vr=vc.pct_change()
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]; p=d.close.astype(float); r=p.pct_change(); v=d.volume.astype(float)
 trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(); relvol=np.log(v/v.rolling(20,min_periods=15).mean()); rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); rv=r.rolling(5,min_periods=4).std(); vv=rv.rolling(20,min_periods=15).std()/rv.rolling(20,min_periods=15).mean()
 # high abnormal-volume losing assets are a conditional short-term reversal candidate
 raw[a]=rev*relvol.abs()
 libs['risk_adjusted_trend_20d'][a]=trend; libs['ravmom_20obs'][a]=trend; libs['relative_volume_participation_20d'][a]=relvol; libs['volnorm_reversal_5obs'][a]=rev; libs['vol_of_vol_cv20'][a]=vv
 # VIX beta, residualized cross-sectionally from own 20d volatility (as admitted definition)
 beta=r.rolling(20,min_periods=15).cov(vr)/vr.rolling(20,min_periods=15).var(); libs['vix_stress_resilience_beta20'][a]=-beta
 for h in H:fw[a,h]=p.shift(-h)/p-1
raw=pd.DataFrame(raw); X={k:pd.DataFrame(v).reindex(raw.index) for k,v in libs.items()}; f=pd.DataFrame(index=raw.index,columns=A,dtype=float)
# Cross-sectional residual against each unique library signal daily. This explicitly isolates the volume-amplified component.
for dt in f.index:
 z=pd.DataFrame({'y':raw.loc[dt]}); z['trend']=X['risk_adjusted_trend_20d'].loc[dt];z['vol']=X['relative_volume_participation_20d'].loc[dt];z['rev']=X['volnorm_reversal_5obs'].loc[dt];z['vov']=X['vol_of_vol_cv20'].loc[dt];z['vixb']=X['vix_stress_resilience_beta20'].loc[dt];z=z.dropna()
 if len(z)>=8:
  q=z[['trend','vol','rev','vov','vixb']]; q=(q-q.mean())/q.std(ddof=0).replace(0,np.nan); q=q.fillna(0)
  M=np.c_[np.ones(len(z)),q.values]; b=np.linalg.lstsq(M,z.y.values,rcond=None)[0];f.loc[dt,z.index]=z.y.values-M@b

def ev(h):
 y=pd.DataFrame({a:fw[a,h] for a in A}).reindex(f.index); out=[];cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:out.append((dt,z.f.corr(z.y,method='spearman')));cov.append(len(z)/15)
 ic=pd.Series(dict(out)); sd=ic.std(ddof=1);return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_std':float(sd),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_hit_ratio':float((ic>0).mean()),'ic_dates':len(ic),'mean_valid_instruments_per_ic_date':float(np.mean(cov)*15),'mean_cross_sectional_coverage':float(np.mean(cov)),'signal_cell_coverage':float(f.notna().mean().mean())}
print('FACTOR orthogonal_volume_amplified_reversal_5d = CS_residual[(-r5/sd5)*abs(log(volume/mean20)) | trend20, relvol20, reversal5, vol_of_vol20, -VIXbeta20]')
print('period',f.index.min().date(),f.index.max().date(),'universe',len(A),'signal_cells',int(f.notna().sum().sum()),'of',f.size)
ms={}
for h in H:
 ic,m=ev(h);ms[h]=m;print('HORIZON',h,json.dumps(m))
 for n,mask in [('2020',ic.index<'2021-01-01'),('2021_2022',(ic.index>='2021-01-01')&(ic.index<'2023-01-01')),('2023_2024',(ic.index>='2023-01-01')&(ic.index<'2025-01-01')),('2025_2026',ic.index>='2025-01-01')]:
  x=ic[mask];print('REGIME',h,n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
r=f.rank(axis=1,pct=True);ts=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('RANK_TURNOVER',round(float(np.mean(ts)),6),'TURNOVER_DATES',len(ts))
mx=0
for n,x in X.items():
 z=pd.concat([f.stack().rename('new'),x.stack().rename('old')],axis=1).dropna();rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'LIBRARY_FILE_COUNT',len([x for x in glob.glob('factors/*.json') if not x.endswith('.bak')]))
print('DECAY',json.dumps({str(h):{'ic':ms[h]['daily_paper_ic'],'icir':ms[h]['daily_paper_icir'],'dates':ms[h]['ic_dates']} for h in H}))
