import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

TODAY='2030-09-05'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Volatility-compression breakout: medium trend rewarded only when recent volatility is compressed
frames={}
for s in assets:
    d=get_stock_daily_data(s,2200)
    if d is None or len(d)<150: d=get_index_daily_data(s,2200)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d
prices=pd.DataFrame({s:d['close'] for s,d in frames.items()}).sort_index()
rets=prices.pct_change()
# lagged signal: 20d trend normalized by 60d volatility, with compression multiplier
r20=prices.pct_change(20); v60=rets.rolling(60).std(); v10=rets.rolling(10).std(); v120=rets.rolling(120).std()
compression=(v10/v120).clip(0.25,2.0)
signal=-(r20/v60)*compression # deliberately test inverse? trend often crowded; evaluate both signs
# use signal as constructed, data available through t, forward from t+1
fwd={h:prices.shift(-h)/prices-1 for h in [1,5,10,20]}
rows=[]
for dt in signal.index:
    x=signal.loc[dt]; n=x.notna()
    if n.sum()<8: continue
    rec={'date':dt,'n':int(n.sum())}
    for h,y in fwd.items():
        z=y.loc[dt,n].dropna(); xx=x.loc[z.index]
        if len(z)>=8 and xx.nunique()>1 and z.nunique()>1: rec[f'ic{h}']=xx.corr(z)
    rows.append(rec)
out=pd.DataFrame(rows).set_index('date')
print('assets',len(frames),'dates',len(out),'avg_n',out.n.mean())
for h in [1,5,10,20]:
    q=out[f'ic{h}'].dropna(); print(h,'IC %.6f ICIR %.6f hit %.4f obs %d'%(q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(),len(q)))
# annual regime stats and signal turnover
print('annual')
for yr,g in out.groupby(out.index.year):
    q=g.ic1.dropna(); print(yr,len(q),'%.5f %.5f'%(q.mean(),q.mean()/q.std(ddof=1) if q.std()>0 else np.nan))
rank=signal.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('coverage',signal.notna().mean().mean(),'turnover',turn)
signal.to_csv('scripts/miner_3_20300905_compression_breakout_signal.csv')
