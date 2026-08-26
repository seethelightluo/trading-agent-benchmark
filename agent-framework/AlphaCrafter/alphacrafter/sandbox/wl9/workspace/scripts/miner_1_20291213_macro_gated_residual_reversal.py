import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# candidate: 10d asset return residual versus cross-asset median, contrarian, scaled by 20d downside vol,
# active only when observation-only VIX is above its trailing 60d median (shock-reversal diversification).
px={}
for s in U:
    d=get_stock_daily_data(s,3000)
    if d is None or len(d)==0: d=get_index_daily_data(s,3000)
    if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
ret=P.pct_change()
vix=get_index_daily_data('VIX',3000)
if vix is None: vix=get_stock_daily_data('VIX',3000)
V=vix.set_index('date')['close'].astype(float).reindex(P.index).ffill() if vix is not None else pd.Series(index=P.index,dtype=float)
rows=[]
for i in range(70,len(P)-10):
    dt=P.index[i]
    if pd.isna(V.loc[dt]) or pd.isna(V.iloc[i-60:i].median()) or V.loc[dt] <= V.iloc[i-60:i].median(): continue
    r10=ret.iloc[i-10:i].sum() # approx cumulative simple sum
    cross=r10.median(); resid=r10-cross
    down=ret.iloc[i-20:i].clip(upper=0).std()
    sig=-(resid/down.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    fwd=ret.iloc[i:i+5].sum()
    z=sig.dropna().rank(pct=True)
    valid=z.index.intersection(fwd.dropna().index)
    if len(valid)>=8:
        rows.append({'date':dt,'ic':z[valid].corr(fwd[valid]),'n':len(valid),'active':1,**{f's_{s}':sig.get(s,np.nan) for s in U}})
R=pd.DataFrame(rows).set_index('date')
print('dates',len(R),'avg_n',R.n.mean() if len(R) else 0,'coverage_obs',len(R)*15/(len(P)*15))
print('IC',R.ic.mean(),'ICIR',R.ic.mean()/R.ic.std(ddof=1),'hit',(R.ic>0).mean(),'turnover',R[[f's_{s}' for s in U]].rank(axis=1,pct=True).diff().abs().mean().mean())
for k in [5,10,20,40]:
  vals=[]
  for i in range(70,len(P)-k):
    dt=P.index[i]
    if dt not in R.index: continue
    r10=ret.iloc[i-10:i].sum(); resid=r10-r10.median(); down=ret.iloc[i-20:i].clip(upper=0).std(); sig=-(resid/down.replace(0,np.nan)); f=ret.iloc[i:i+k].sum(); z=sig.rank(pct=True); v=z.dropna().index.intersection(f.dropna().index)
    if len(v)>=8: vals.append(z[v].corr(f[v]))
  a=np.array(vals); print('h',k,'dates',len(a),'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
# artifact
R.to_csv('scripts/miner_1_20291213_macro_gated_residual_reversal_signal.csv')
