"""One idea: inverse peer-relative downside semideviation conditional on elevated cross-asset dispersion."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; close={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
    close[a]=pd.to_numeric(d.sort_values('date').set_index('date')['close'],errors='coerce')
p=pd.DataFrame(close); r=p.pct_change(); peer=r.median(axis=1)
# A session is stressed only if its cross-asset return dispersion exceeds its trailing 60-session median.
disp=r.std(axis=1); elevated=disp.gt(disp.rolling(60,min_periods=40).median())
# On elevated-dispersion sessions retain only an asset's losses versus the peer median.
rel_loss=r.sub(peer,axis=0).clip(upper=0).where(elevated,0.0)
# Negation rewards assets with shallower relative losses during dispersed stress; one-day lag prevents look-ahead.
f=-np.sqrt(rel_loss.pow(2).rolling(60,min_periods=40).mean()).shift(1)
f=f.sub(f.median(axis=1),axis=0); horizons=[1,5,10,20]; cutoff=p.dropna(how='all').index.max(); forward={h:p.shift(-h)/p-1 for h in horizons}
def evaluate(h, span=None):
    x=f if span is None else f.loc[span[0]:span[1];]; y=forward[h].reindex(x.index); vals=[]; breadth=[]
    for dt in x.index:
        q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
        if len(q)>=8:
            v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
            if np.isfinite(v): vals.append(v); breadth.append(len(q))
    if not vals:return dict(dates=0)
    z=np.asarray(vals); sd=z.std(ddof=1)
    return dict(dates=len(z),ic=round(float(z.mean()),6),icir=round(float(z.mean()/sd),6),hit=round(float((z>0).mean()),4),mean_n=round(float(np.mean(breadth)),2),min_n=int(min(breadth)))
print('FACTOR inverse_peer_relative_downside_semideviation_elevated_dispersion_60 cutoff',cutoff.date(),'assets',len(assets))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2),'elevated_session_rate',round(float(elevated.mean()),4))
for h in horizons: print('H',h,evaluate(h))
for label,span in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-12-31')),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',label,evaluate(10,span))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
