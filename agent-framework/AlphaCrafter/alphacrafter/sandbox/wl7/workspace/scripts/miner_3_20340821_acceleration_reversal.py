import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try: d=get_index_daily_data(s, days=5200)
    except Exception:
        try: d=get_stock_daily_data(s, days=5200)
        except Exception: d=None
    if d is not None and len(d)>=150:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(frames).sort_index().ffill()
# Explicit sign inversion of prior failed acceleration: short-term strength relative to medium-term strength.
f=px.pct_change(20)-px.pct_change(60)
print('loaded',len(frames),'dates',len(px),'instruments',len(U))
allmetrics={}
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in px.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); ns.append(ok.sum())
 a=np.asarray(vals); ic=np.nanmean(a); icir=ic/np.nanstd(a,ddof=1)
 allmetrics[h]=(ic,icir,len(a),np.mean(ns))
 print(f'H{h}: IC {ic:.6f} ICIR {icir:.6f} hit {(a>0).mean():.4f} dates {len(a)} avgN {np.mean(ns):.2f}')
 for yr in [(2020,2022),(2023,2026),(2027,2030),(2031,2034)]:
  z=np.array([v for v,d in zip(vals,dates) if yr[0]<=d.year<=yr[1]])
  if len(z)>1: print(f'  {yr}: n={len(z)} IC={np.mean(z):.6f} ICIR={np.mean(z)/np.std(z,ddof=1):.6f}')
fr=px.shift(-10)/px-1; vals=[]
for dt in px.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: vals.append((dt,spearmanr(x[ok],y[ok]).statistic))
for n in [180,500,750]:
 z=np.array([v for _,v in vals[-n:]]); print(f'recent{n} H10 IC {np.mean(z):.6f} ICIR {np.mean(z)/np.std(z,ddof=1):.6f} n {len(z)}')
ranks=f.rank(axis=1,pct=True); print(f'coverage {f.notna().sum(axis=1).mean()/len(U):.6f} turnover {ranks.diff().abs().mean(axis=1).mean():.6f} rows {f.stack().size}')
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20340821_acceleration_reversal_signal.csv',index=False)
print('artifact scripts/miner_3_20340821_acceleration_reversal_signal.csv')
