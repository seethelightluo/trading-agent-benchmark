"""One idea: downside-dispersion shock gated peer-relative reversal, 20-observation signal."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C);r=P.pct_change();med=r.median(axis=1);cut=P.dropna(how='all').index.max(); H=[1,5,10,20]
# Only downside cross-sectional deviations enter dispersion; a high and rising state gates a slow peer-relative reversal.
down=r.sub(med,axis=0).where(r.sub(med,axis=0)<0,0).abs().median(axis=1)
shock=(down>down.rolling(60,min_periods=45).quantile(.75))&(down>down.shift(5))
raw=-r.sub(med,axis=0).rolling(5,min_periods=4).sum()/r.rolling(20,min_periods=15).std()
event=raw.where(shock,axis=0); f=event.rolling(20,min_periods=5).mean();f=f.sub(f.median(axis=1),axis=0)
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=(P.shift(-h)/P-1).reindex(x.index);z=[];ns=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z);sd=z.std(ddof=1)
 return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/sd),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))} if len(z) else {}
print('FACTOR downside_dispersion_shock_peer_reversal_20 cutoff',cut.date(),'assets',len(A));print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),5))
for h in H:print('H',h,ev(h))
for n,s in [('2020_24',('2020-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01',str(cut.date())) )]:print('REGIME20',n,ev(20,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'EVENT_RATE',round(float(shock.mean()),4))
# Correlation evidence against closest admitted signals (standard and downside-dispersion variants).
vol=r.rolling(20,min_periods=15).std(); disp=r.sub(med,axis=0).abs().median(axis=1); stdshock=(disp>disp.rolling(60,min_periods=45).quantile(.75))&(disp>disp.shift(5)); old=(-r.sub(med,axis=0).rolling(5,min_periods=4).sum()/vol).where(stdshock,axis=0).rolling(20,min_periods=5).mean();old=old.sub(old.median(axis=1),axis=0)
# broad peer reversal and short vol-normalized reversal are also admitted-library comparators
S={'dispersion_shock_peer_reversal_20':old,'peer_relative_reversal_5':-r.sub(med,axis=0).rolling(5,min_periods=4).sum()/vol,'volnorm_reversal_5':-P.pct_change(5)/r.rolling(5,min_periods=4).std(),'risk_adjusted_momentum_20':P.pct_change(20)/vol,'inverse_idio_vol_20':-r.sub(med,axis=0).rolling(20,min_periods=15).std()}
mx=0
for n,g in S.items():
 q=pd.concat([f.stack(),g.stack()],axis=1).dropna();rho=float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);print('LIBCORR',n,'cells',len(q),'rho',round(rho,6));mx=max(mx,abs(rho))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'COMPARATORS',len(S))
print('NOTE: full library admission requires correlation evidence versus every admitted signal; this candidate is rejected unless this screen is clearly below limit and a complete audit is feasible.')
