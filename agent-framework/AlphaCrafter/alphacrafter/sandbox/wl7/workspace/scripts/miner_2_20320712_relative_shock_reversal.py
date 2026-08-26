import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(sym, days=5000):
    d=get_stock_daily_data(sym,days)
    if d is None or len(d)<100: d=get_index_daily_data(sym,days)
    return d
raw={s:get(s) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in raw.items() if d is not None}).sort_index()
# Candidate: relative 10d shock reversal, scaled by idiosyncratic 30d volatility.
logp=np.log(px)
r10=logp-logp.shift(10)
cs_med=r10.median(axis=1)
resid=r10.sub(cs_med,axis=0)
vol=logp.diff().rolling(30).std()
sig=-resid/vol
# forward returns, cross sectional IC on each date; data is lag-safe by construction
out=[]
for h in [1,5,10,20]:
    fwd=logp.shift(-h)-logp
    vals=[]
    for dt in sig.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
    q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
    ic=q.ic.dropna()
    mean=ic.mean(); sd=ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
    print(f'H{h}: dates={len(q)} avgN={q.n.mean():.2f} IC={mean:.6f} ICIR={icir:.6f} hit={(ic>0).mean():.4f}')
    if h==10:
        thirds=np.array_split(ic,3); print(' thirds',*[f'{a.mean():.6f}' for a in thirds])
# coverage and rank turnover
valid=sig.notna().sum(axis=1); print('signal_dates',len(sig),'coverage',valid.sum()/(len(sig)*len(U)),'avgN',valid.mean())
r=sig.rank(axis=1,pct=True); turn=(r.diff().abs().sum(axis=1)/2).mean(); print('rank_turnover',turn)
sig.to_csv('scripts/miner_2_20320712_relative_shock_reversal_signal.csv',index_label='date')
print('period',px.index.min(),px.index.max(),'assets',len(px.columns))
