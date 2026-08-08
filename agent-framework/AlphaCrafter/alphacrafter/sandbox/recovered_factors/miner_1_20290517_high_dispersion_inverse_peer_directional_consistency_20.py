"""One idea: high-dispersion inverse peer-relative directional consistency (20 sessions)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; cut=pd.Timestamp('2029-05-16'); C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d[d.date<=cut].sort_values('date').set_index('date')['close'],errors='coerce')
P=pd.DataFrame(C);R=P.pct_change(); rel=R.sub(R.median(axis=1),axis=0)
# A high value is an intentionally contrarian score: persistent peer-relative losers after a high-dispersion state.
path=rel.rolling(20,min_periods=16).sum()/rel.abs().rolling(20,min_periods=16).sum()
disp=R.std(axis=1); elevated=disp.rolling(20,min_periods=16).mean()>disp.rolling(252,min_periods=126).median()
f=(-path).where(elevated, np.nan).shift(1); f=f.sub(f.median(axis=1),axis=0)
H=[1,5,10,20];FW={h:P.shift(-h)/P-1 for h in H}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=FW[h].reindex(x.index); z=[]; nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(nn)),2),'min_n':int(min(nn))}
print('FACTOR high_dispersion_inverse_peer_directional_consistency_20 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2),'elevated_share',round(float(elevated.mean()),4))
for h in H:print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cut.date()))),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]: print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
