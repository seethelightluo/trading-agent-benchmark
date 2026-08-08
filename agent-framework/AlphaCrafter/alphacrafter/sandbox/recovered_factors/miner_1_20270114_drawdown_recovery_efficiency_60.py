"""miner_1: 60d drawdown recovery efficiency. Validation through 2027-01-13 only."""
import numpy as np, pandas as pd, glob, json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2027-01-13'); H=[1,5,10,20]
C={};R={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date')
 C[a]=d.close.replace(0,np.nan); R[a]=C[a].pct_change(); V[a]=d.volume.replace(0,np.nan)
# Recovery efficiency: current rebound from trailing 60d trough divided by prior peak-to-trough drawdown, clipped; high means a substantially repaired drawdown.
def calc(c):
 peak=c.rolling(60,min_periods=45).max(); trough=c.rolling(60,min_periods=45).min()
 return ((c/trough-1)/(peak/trough-1).replace(0,np.nan)).clip(0,1)
factor=pd.DataFrame({a:calc(C[a]) for a in A}); fwd={h:pd.DataFrame({a:C[a].shift(-h)/C[a]-1 for a in A}) for h in H}
def ev(h,span=None):
 f=factor if span is None else factor.loc[span[0]:span[1]]; y=fwd[h] if span is None else fwd[h].loc[span[0]:span[1]]; z=[]; ns=[]; const=0
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
   else:const+=1
 z=np.array(z)
 return {'dates':len(z),'dropped_constant':const,'ic':float(z.mean()),'icir':float(z.mean()/z.std(ddof=1)),'hit':float((z>0).mean()),'mean_names':float(np.mean(ns)),'min_names':int(np.min(ns))} if len(z) else {'dates':0}
print('FACTOR drawdown_recovery_efficiency_60 cutoff',cutoff.date());print('CELLS',int(factor.notna().sum().sum()),'/',factor.size,'coverage',float(factor.notna().stack().mean()))
for h in H:print('H',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027',('2027-01-01','2027-01-13'))]:print('REGIME_10',n,ev(10,s))
print('TURNOVER',float(factor.rank(axis=1,pct=True).diff().abs().stack().mean()))
# Signal-correlations versus every admitted implementation; VIX conditional factors are reconstructed only if VIX data is available, otherwise evidence deliberately fails.
def quiet(a):return (C[a].pct_change(20).abs()/R[a].abs().rolling(20,min_periods=15).sum())*(1-R[a].rolling(20,min_periods=15).std().rolling(60,min_periods=40).apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1],raw=False))
lib={'relative_volume_participation':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}),'risk_adjusted_trend':pd.DataFrame({a:(C[a]/C[a].shift(20)-1)/R[a].rolling(20,min_periods=15).std() for a in A}),'ravmom_20':pd.DataFrame({a:(C[a]/C[a].shift(20)-1)/R[a].abs().rolling(20,min_periods=15).mean() for a in A}),'volnorm_reversal_5':pd.DataFrame({a:-(C[a]/C[a].shift(5)-1)/R[a].rolling(5,min_periods=4).std() for a in A}),'volscaled_reversal_1':pd.DataFrame({a:-R[a]/R[a].rolling(20,min_periods=15).std() for a in A}),'quiet_path':pd.DataFrame({a:quiet(a) for a in A})}
# load persisted expressions not executable; reconstruct VIX signals from documented JSON parameter/expression if available
for fp in glob.glob('factors/*.json'):
 d=json.load(open(fp)); fid=d.get('factor_id','')
 if 'vix_' in fid: print('VIX_LIBRARY_MEMBER',fid,'correlation evidence unavailable in candidate script')
mx=0
for n,x in lib.items():
 q=pd.concat([factor.stack(),x.stack()],axis=1).dropna();rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);mx=max(mx,abs(rho));print('LIBCORR',n,'cells',len(q),'rho',rho)
print('MAX_ABS_RECONSTRUCTED_LIBRARY_CORRELATION',mx)
