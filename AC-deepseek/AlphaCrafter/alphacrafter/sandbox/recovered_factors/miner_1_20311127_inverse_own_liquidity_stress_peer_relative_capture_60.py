"""One factor idea: inverse own-liquidity-stress peer-relative capture (60 sessions)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={};VV={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce'); VV[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C); V=pd.DataFrame(VV).reindex(P.index); r=P.pct_change(); peer=r.sub(r.median(axis=1),axis=0)
# Each asset's volume stress is relative only to its own history, preventing cross-market volume scale effects.
lv=np.log(V/V.rolling(20,min_periods=15).mean())
stress=lv.lt(lv.rolling(60,min_periods=40).quantile(.20).shift(1))
# Mean relative return earned on own low-liquidity dates; invert because thin-market relative winners tend to normalize.
f=pd.DataFrame({a:-peer[a].where(stress[a]).rolling(60,min_periods=12).mean() for a in A})
f=f.sub(f.median(axis=1),axis=0).shift(1); fw={h:P.shift(-h)/P-1 for h in [1,5,10,20]}; cutoff=P.dropna(how='all').index.max()
def stats(h,sl=None):
 x=f if sl is None else f.loc[sl[0]:sl[1]]; z=[]; nn=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8: z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);nn.append(len(q))
 z=np.array(z)
 return dict(dates=len(z),ic=round(float(z.mean()),6),icir=round(float(z.mean()/z.std(ddof=1)),6),hit=round(float((z>0).mean()),6),breadth=round(float(np.mean(nn)),3),min_breadth=int(min(nn)))
print('FACTOR inverse_own_liquidity_stress_peer_relative_capture_60','CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'COVERAGE',round(float(f.notna().stack().mean()),6),'TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'DISPERSION',round(float(f.std(axis=1).mean()),6))
for h in [1,5,10,20]: print('H',h,stats(h))
for name,sl in [('2025_2026',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',name,stats(10,sl))
