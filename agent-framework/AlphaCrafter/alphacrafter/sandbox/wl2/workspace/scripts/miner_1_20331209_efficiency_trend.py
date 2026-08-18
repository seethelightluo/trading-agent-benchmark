import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=get_stock_daily_data(s,days=4100)
    if d is None or len(d)<100: d=get_index_daily_data(s,days=4100)
    if d is not None: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
r=np.log(P).diff()
# Efficiency-adjusted trend: directional 20-session move divided by path length;
# lag one day so the signal uses only completed data.
ret20=np.log(P/P.shift(20))
path20=r.abs().rolling(20,min_periods=18).sum()
eff=(ret20/path20).shift(1)
# cross-sectional rank, preserving directional information
sig=eff.rank(axis=1,pct=True).sub(.5)
rows=[]; signals=[]
for h in [5,10,20]:
    f=np.log(P.shift(-h)/P)
    vals=[]
    for dt in sig.index:
        z=sig.loc[dt]; y=f.loc[dt]
        ok=z.notna()&y.notna()
        if ok.sum()>=8:
            vals.append((dt,z[ok].corr(y[ok]),int(ok.sum())))
    q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
    ic=q.ic.mean(); sd=q.ic.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
    print('H',h,'dates',len(q),'avg_n',q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(ic,icir,(q.ic>0).mean()))
    if h==10: q.to_csv('scripts/miner_1_20331209_efficiency_trend_10d_ic.csv')
# coverage and turnover proxy
valid=sig.notna(); coverage=valid.sum().sum()/(len(sig)*len(U))
turn=sig.diff().abs().mean().mean()
print('assets',len(px),'dates',len(P),'coverage %.6f turnover %.6f'%(coverage,turn))
# recoverable signal artifact
sig.to_csv('scripts/miner_1_20331209_efficiency_trend_signal.csv',index_label='date')
