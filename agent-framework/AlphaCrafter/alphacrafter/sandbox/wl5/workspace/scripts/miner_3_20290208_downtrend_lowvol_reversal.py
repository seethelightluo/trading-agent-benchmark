import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2029-02-07'
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut].astype(float); r=P.pct_change()
vol=r.rolling(20,min_periods=15).std(); base=(-P.pct_change(5)/(vol*np.sqrt(5)))
trend=P.pct_change(20)
# Downside-trend-conditioned volatility-scaled reversal: give larger reversal score to assets that also lost over 20d.
F=(base*(1+0.50*(trend<0))).replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_3_20290208_downtrend_lowvol_reversal_signal.csv')
def run(h=10,a=None,b=None):
 v=[]; cov=[]; turns=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if a and not(a<=d<=b): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   v.append(spearmanr(z.f,z.y).statistic);cov.append(len(z)/15)
 x=np.asarray(v); return len(x),float(np.mean(cov)),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)),float(np.mean(x>0))
print('range',P.index.min().date(),P.index.max().date(),'assets',len(U),'rows',len(P))
for h in [5,10,20]: print('horizon',h,run(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-02-07'),('2028-08-01','2029-02-07')]: print('regime',a,b,run(10,a,b))
# signal turnover, measured rank-order changes on consecutive valid dates
ranks=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 a,b=ranks.iloc[i-1],ranks.iloc[i]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8: turns.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('rank_turnover',float(np.mean(turns)),'turnover_obs',len(turns),'coverage',float(F.notna().mean().mean()))
# max corr with existing latest signal artifact where aligned
for fn in ['scripts/miner_3_20290125_lowvol_reversal_signal.csv','scripts/miner_3_20281228_trend_damped_dispersion_reversal_5d_signal.csv']:
 try:
  old=pd.read_csv(fn,index_col=0,parse_dates=True); a,b=F.align(old,join='inner'); x=pd.concat([a.stack(),b.stack()],axis=1).dropna(); print('corr',fn,float(x.corr().iloc[0,1]),len(x))
 except Exception as e: print('corr_error',fn,e)
