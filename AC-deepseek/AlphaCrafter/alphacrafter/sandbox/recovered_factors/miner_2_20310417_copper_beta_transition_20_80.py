"""Miner 2: copper beta transition, 20 versus 80 completed sessions."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C).sort_index(); R=P.pct_change(); cr=R['COPPER']; cutoff=P.dropna(how='all').index.max()
def beta(x,y,w,minp): return x.rolling(w,min_periods=minp).cov(y)/y.rolling(w,min_periods=minp).var()
b20=pd.DataFrame({a:beta(R[a],cr,20,15) for a in A}); b80=pd.DataFrame({a:beta(R[a],cr,80,55) for a in A})
# Rising copper sensitivity, standardized cross-sectionally; shifted so only prior completed data informs prediction.
f=(b20-b80).sub((b20-b80).median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
def stat(h,sl=None):
 x=f if sl is None else f.loc[sl[0]:sl[1]]; y=fw[h].reindex(x.index); z=[]; n=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);n.append(len(q))
 z=np.array(z)
 if not len(z):return {'ic_dates':0}
 return {'ic_dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit_ratio':round(float((z>0).mean()),6),'mean_valid_names':round(float(np.mean(n)),3),'min_valid_names':int(min(n))}
print('FACTOR copper_beta_transition_20_80 CUTOFF',cutoff.date(),'INSTRUMENTS',len(A),'DATES',len(P))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'COVERAGE',round(float(f.notna().stack().mean()),6))
for h in fw:print('HORIZON',h,stat(h))
for label,sl in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-12-31')),('2029_current',('2029-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',label,stat(10,sl))
print('TURNOVER_RANK_CHANGE',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
print('CROSS_SECTIONAL_STD',round(float(f.std(axis=1).mean()),6))
