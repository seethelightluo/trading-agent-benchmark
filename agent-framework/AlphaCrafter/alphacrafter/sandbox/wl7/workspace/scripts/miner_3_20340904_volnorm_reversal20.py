import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=None
    try: d=get_index_daily_data(s, days=5200)
    except Exception: pass
    if d is None:
        try: d=get_stock_daily_data(s, days=5200)
        except Exception: pass
    if d is not None and len(d)>=120:
        d=d.copy(); d.date=pd.to_datetime(d.date); frames[s]=d.set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in frames.items()}).sort_index().ffill()
# Volatility-normalized medium-term reversal: recent drawdown scaled by trailing realized risk.
ret=px.pct_change(); vol=ret.rolling(20,min_periods=15).std()*np.sqrt(20)
f=-px.pct_change(20)/vol.replace(0,np.nan)
print('loaded',len(frames),'dates',len(px),'instruments',len(U))
for h in [1,5,10,20]:
 vals=[]; dates=[]; ns=[]
 fr=px.shift(-h)/px-1
 for dt in px.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); ns.append(ok.sum())
 a=np.asarray(vals); ic=np.nanmean(a); icir=ic/np.nanstd(a,ddof=1)
 print(f'H{h}: IC {ic:.6f} ICIR {icir:.6f} hit {(a>0).mean():.4f} dates {len(a)} avgN {np.mean(ns):.2f}')
 for yr in [(2020,2022),(2023,2026),(2027,2030),(2031,2034)]:
  z=np.array([v for v,d in zip(vals,dates) if yr[0]<=d.year<=yr[1]])
  if len(z)>1: print(f'  {yr}: n={len(z)} IC={np.mean(z):.6f} ICIR={np.mean(z)/np.std(z,ddof=1):.6f}')
ranks=f.rank(axis=1,pct=True)
print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',ranks.diff().abs().mean(axis=1).mean(),'rows',f.stack().size)
out=f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'})
out.to_csv('scripts/miner_3_20340904_volnorm_reversal20_signal.csv',index=False)
print('artifact scripts/miner_3_20340904_volnorm_reversal20_signal.csv')
