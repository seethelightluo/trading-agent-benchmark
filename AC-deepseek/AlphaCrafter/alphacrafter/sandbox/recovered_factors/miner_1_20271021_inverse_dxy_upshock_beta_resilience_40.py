"""Single idea: inverse DXY-up-shock beta resilience, 40 observations."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=[1,5,10,20]; cutoff=pd.Timestamp('2027-10-20')
C={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date');C[a]=d.close.replace(0,np.nan);V[a]=d.volume.replace(0,np.nan)
P=pd.DataFrame(C);r=P.pct_change();m=r.median(axis=1); vol=r.rolling(20,min_periods=15).std()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.pct_change()
# High value means low/negative sensitivity to daily DXY changes, measured only on prior DXY-up shock days.
shock=(dxy>dxy.rolling(60,min_periods=40).quantile(.65).shift(1))
def cb(x,y,cond,w=40):
 z=pd.concat([x.rename('x'),y.rename('y'),cond.rename('c')],axis=1).where(lambda q:q.c).drop(columns='c');return z.x.rolling(w,min_periods=12).cov(z.y)/z.y.rolling(w,min_periods=12).var()
f=pd.DataFrame({a:-cb(r[a],dxy,shock) for a in A}).reindex(P.index)
fw={h:P.shift(-h).div(P)-1 for h in H}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=fw[h].loc[x.index]; z=[];ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.asarray(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(np.min(ns))}
print('FACTOR inverse_dxy_upshock_beta_resilience_40 cutoff',cutoff.date(),'assets',len(A));print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),5))
for h in H:print('H',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027YTD',('2027-01-01','2027-10-20'))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
# reconstructed active-library signals, pooled signal Spearman evidence
other=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
def beta(x,y,side=None,w=40):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1)
 if side=='down':z=z.where(z.y<0)
 if side=='up':z=z.where(z.y>0)
 return z.x.rolling(w,min_periods=12).cov(z.y)/z.y.rolling(w,min_periods=12).var()
trend=(P/P.shift(20)-1)/vol; eff=P.pct_change(20).abs()/r.abs().rolling(20,min_periods=15).sum(); corr20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(other[a]) for a in A})
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date').close.pct_change(); sg=pd.Series(np.where(vix/vix.shift(20)-1>0,-1.,1.),index=vix.index)
q20=r.rolling(60,min_periods=40).quantile(.2).shift(1)
S={'ravmom':trend,'risk_adj_trend':trend,'volnorm_rev5':-P.pct_change(5)/r.rolling(5,min_periods=4).std(),'volscaled_rev1':-r/vol,'rel_volume':pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A}),'stable_liq':-pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()).rolling(20,min_periods=15).std() for a in A}),'quiet_path':eff*(1-vol.rolling(60,min_periods=40).rank(pct=True)),'gradual_vol_contract':trend*np.tanh((-np.log(vol/r.rolling(40,min_periods=15).std())).clip(-2,2)),'inverse_lag1':-pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A}),'vol_transition':-pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A})*np.log(r.rolling(5,min_periods=4).std()/vol).clip(-2,2),'inverse_idio':-r.sub(m,axis=0).rolling(20,min_periods=15).std(),'low_commonality':-pd.DataFrame({a:r[a].rolling(40,min_periods=25).corr(other[a]) for a in A}),'common_expand':corr20.rolling(20,min_periods=15).mean()-corr20.shift(20).rolling(20,min_periods=15).mean(),'downside_beta':pd.DataFrame({a:beta(r[a],m,'down') for a in A}),'beta_asym':pd.DataFrame({a:beta(r[a],m,'down',60)-beta(r[a],m,'up',60) for a in A}),'downside_event':r.sub(m,axis=0).where((m<m.rolling(60,min_periods=40).quantile(.35)).shift(1),axis=0).rolling(40,min_periods=12).median(),'lower_tail':-r.lt(q20).where(q20.notna()).rolling(40,min_periods=25).mean(),'skew60':r.rolling(60,min_periods=40).skew(),'vix_trend':trend.mul(sg,axis=0),'vix_up_beta':pd.DataFrame({a:-beta(r[a],vix,'up') for a in A})}
mx=0;who=''
for n,g in S.items():
 z=pd.concat([f.stack(),g.stack()],axis=1).dropna();rho=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);print('LIBCORR',n,'cells',len(z),'rho',round(rho,6))
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'N_FACTORS',len(S))
