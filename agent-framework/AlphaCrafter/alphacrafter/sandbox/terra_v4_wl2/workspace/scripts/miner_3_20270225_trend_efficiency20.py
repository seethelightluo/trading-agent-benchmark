import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:get_stock_daily_data(a,days=4000) for a in U}
px={a:d.set_index('date')['close'].astype(float) for a,d in D.items() if d is not None and len(d)>100}
vol={a:d.set_index('date')['volume'].astype(float) for a,d in D.items() if d is not None and len(d)>100}
C=pd.DataFrame(px).sort_index(); R=C.pct_change()
# Trend efficiency: directional 20d return divided by total absolute path, cross-sectional rank-compatible.
ret=C/C.shift(20)-1
path=R.abs().rolling(20).sum()
f=ret/path
# point-in-time factor; forward returns at horizons
for h in [1,3,5,10]:
    fr=C.shift(-h)/C-1
    vals=[]; dates=[]; ns=[]
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
            dates.append(dt); ns.append(len(z))
    s=pd.Series(vals,index=dates).replace([np.inf,-np.inf],np.nan).dropna()
    print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',np.mean(s>0),'sd',s.std(ddof=1))
# turnover rank changes daily
rank=f.rank(axis=1,pct=True)
turn=(rank.diff().abs().mean(axis=1)).mean()
print('coverage',f.notna().mean().mean(),'turnover',turn,'assets',len(px),'dates',len(C))
# regimes
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-06-30'),('2026-07-01','2027-02-23')]:
    s=[]
    for dt in f.index:
      if str(dt)>=lo and str(dt)<=hi:
       z=pd.concat([f.loc[dt],(C.shift(-1)/C-1).loc[dt]],axis=1).dropna()
       if len(z)>=8:s.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    s=pd.Series(s).dropna(); print('REG',lo,hi,len(s),s.mean(),s.mean()/s.std(ddof=1) if len(s)>1 else np.nan)
# artifact for admission horizon 1
out=pd.DataFrame(f.stack(),columns=['signal']);out.index.names=['date','symbol'];out.to_csv('../persistent/factor_signals_miner_3_20270225_trend_efficiency20.csv')
