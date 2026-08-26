import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=None
    try: d=get_index_daily_data(s, days=5200)
    except (FileNotFoundError, Exception):
        try: d=get_stock_daily_data(s, days=5200)
        except Exception: d=None
    if d is not None and len(d)>=150:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.set_index('date')['close'].astype(float)
print('loaded',len(frames),sorted(frames))
px=pd.DataFrame(frames).sort_index().ffill(); r20=px.pct_change(20); r60=px.pct_change(60); f=r60-r20
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in px.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic); dates.append(dt); ns.append(ok.sum())
 a=np.array(vals); print(f'H{h}: IC {np.nanmean(a):.6f} ICIR {np.nanmean(a)/np.nanstd(a,ddof=1):.6f} hit {(a>0).mean():.4f} dates {len(a)} avgN {np.mean(ns):.2f}')
 for yr in [(2020,2022),(2023,2026),(2027,2030),(2031,2034)]:
  z=np.array([v for v,d in zip(vals,dates) if yr[0]<=d.year<=yr[1]])
  if len(z): print(f'  {yr}: n={len(z)} IC={np.mean(z):.6f} ICIR={np.mean(z)/np.std(z,ddof=1):.6f}')
fr=px.shift(-10)/px-1; vals=[]
for dt in px.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: vals.append((dt,spearmanr(x[ok],y[ok]).statistic))
for n in [180,500,750]:
 z=np.array([v for _,v in vals[-n:]]); print(f'recent{n} H10 IC {np.mean(z):.6f} ICIR {np.mean(z)/np.std(z,ddof=1):.6f} n {len(z)}')
ranks=f.rank(axis=1,pct=True); print(f'coverage {f.notna().sum(axis=1).mean()/len(U):.6f} turnover {ranks.diff().abs().mean(axis=1).mean():.6f} rows {f.stack().size} dates {len(px)} instruments {len(U)}')
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20340807_acceleration_signal.csv',index=False)
print('artifact scripts/miner_3_20340807_acceleration_signal.csv')
