import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
macro={'DXY','USDCNY','USDJPY','EURUSD','VIX'}
def fetch(s):
    for f in (get_stock_daily_data,get_index_daily_data):
        try:
            x=f(s,days=3000)
            if x is not None and len(x): return x
        except Exception: pass
    return None
raw={s:fetch(s) for s in U}
prices=pd.DataFrame({s:x.set_index('date')['close'] for s,x in raw.items() if x is not None}).sort_index()
rets=prices.pct_change()
# Volatility asymmetry: signed difference between upside and downside RMS, scaled by total vol.
# Uses only trailing completed observations at each signal date.
pos=rets.clip(lower=0).rolling(20,min_periods=15).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
neg=(-rets.clip(upper=0)).rolling(20,min_periods=15).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
tot=rets.rolling(60,min_periods=40).std()
factor=(pos-neg)/(tot+1e-12)
# cross-sectional rank not needed for IC; forward close-to-close returns
outs=[]
for h in [1,5,10]:
    vals=[]; dates=[]
    fwd=prices.shift(-h)/prices-1
    for d in factor.index:
        a=factor.loc[d]; b=fwd.loc[d]
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d)
    q=pd.Series(vals,index=pd.to_datetime(dates))
    print(f'h={h} dates={len(q)} avg_n={np.nanmean([pd.concat([factor.loc[d],fwd.loc[d]],axis=1).dropna().shape[0] for d in dates]):.2f} coverage={np.isfinite(factor.loc[dates]).sum().sum()/(len(dates)*len(U)):.4f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
    for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028'),('2029','2030')]:
        t=q[(q.index>=lo)&(q.index<=hi)]
        if len(t): print(f'  {lo}-{hi}: n={len(t)} IC={t.mean():.6f}')
# turnover: rank top/bottom signal changes, average absolute rank movement
r=factor.rank(axis=1,pct=True)
turn=(r.diff().abs().mean(axis=1)).dropna().mean()
print(f'turnover_rank_abs_mean={turn:.6f} instruments={len(prices.columns)} dates={len(prices)}')
