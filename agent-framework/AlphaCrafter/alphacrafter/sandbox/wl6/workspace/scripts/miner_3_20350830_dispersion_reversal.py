import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    x=get_stock_daily_data(s, days=5000)
    if x is None or len(x)==0: x=get_index_daily_data(s, days=5000)
    return x[['date','close']].drop_duplicates('date').set_index('date')['close'] if x is not None else None
px={s:fetch(s) for s in U}; px={s:x for s,x in px.items() if x is not None}
C=pd.DataFrame(px).sort_index().ffill(); C=C.loc[C.index<=pd.Timestamp('2035-08-29')]
r=np.log(C/C.shift(1)); ret20=C/C.shift(20)-1; med=ret20.median(axis=1); resid=ret20.sub(med,axis=0)
disp=r.rolling(20).std().median(axis=1); threshold=disp.rolling(252,min_periods=126).median()
sig=-resid.where(disp>threshold)
# Persist recoverable signal artifact for deterministic post-Miner screening.
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20350830_dispersion_reversal_signal.csv',index=False)
for h in [5,10,20,40]:
    fwd=C.shift(-h)/C-1; vals=[]; ninst=[]
    for d in sig.index:
        z=sig.loc[d]; y=fwd.loc[d]; ok=z.notna()&y.notna()
        if ok.sum()>=8: vals.append(z[ok].corr(y[ok],method='spearman')); ninst.append(ok.sum())
    a=pd.Series(vals).dropna(); ic=a.mean(); icir=ic/a.std(ddof=1)*np.sqrt(len(a))
    print(f'h={h} dates={len(a)} avg_inst={np.mean(ninst):.3f} IC={ic:.8f} ICIR={icir:.8f} hit={(a>0).mean():.4f}')
valid=sig.notna().sum().sum()/(len(sig)*len(U)); turnover=sig.rank(axis=1,pct=True).diff().abs().mean().mean()
print(f'coverage={valid:.6f} turnover={turnover:.6f} instruments={len(U)} dates={len(sig)} end={C.index.max().date()}')
