import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; W=100
frames={}
for s in UNIV:
    try: d=get_stock_daily_data(s,days=3000)
    except Exception:
        try: d=get_index_daily_data(s,days=3000)
        except Exception: d=None
    if d is not None and len(d)>W+70:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.set_index('date')['close'].astype(float).sort_index()
px=pd.concat(frames,axis=1).sort_index(); r=px.pct_change(); sig=(r>0).rolling(W,min_periods=80).mean().shift(1); rows=[]
for dt in sig.index:
 x=sig.loc[dt].dropna().rename('signal')
 if len(x)>=8: rows.append((dt,x))
for h in [5,10,20,40,60]:
 ics=[]; ns=[]
 for dt,x in rows:
  fut=px.loc[px.index>dt].reindex(columns=x.index).head(h)
  if len(fut)<h: continue
  fr=fut.iloc[-1]/px.loc[dt,x.index]-1; z=pd.concat([x,fr.rename('f')],axis=1).dropna()
  if len(z)>=8 and z['signal'].nunique()>1: ics.append(z['signal'].corr(z['f'],method='spearman')); ns.append(len(z))
 a=np.array(ics); print(f'H={h} dates={len(a)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={np.nanmean(a):.6f} ICIR={np.nanmean(a)/np.nanstd(a,ddof=1):.6f} hit={np.mean(a>0):.4f}')
 if h==20:
  out=[]
  for dt,x in rows:
   fut=px.loc[px.index>dt].reindex(columns=x.index).head(h)
   if len(fut)<h: continue
   fr=fut.iloc[-1]/px.loc[dt,x.index]-1
   for s,v in x.items():
    if pd.notna(fr.get(s)): out.append({'date':dt,'symbol':s,'signal':v,'forward_return':fr[s]})
  pd.DataFrame(out).to_csv('scripts/miner_2_20310220_trend_persistence_signal.csv',index=False)
ranks=sig.rank(axis=1,pct=True); print(f'TURNOVER_PROXY={ranks.diff().abs().mean(axis=1).mean():.6f} total_dates={len(rows)} instruments={len(frames)}')
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31')]:
 vals=[]
 for dt,x in rows:
  if not(pd.Timestamp(a)<=dt<=pd.Timestamp(b)): continue
  fut=px.loc[px.index>dt].reindex(columns=x.index).head(20)
  if len(fut)<20: continue
  fr=fut.iloc[-1]/px.loc[dt,x.index]-1; z=pd.concat([x,fr.rename('f')],axis=1).dropna()
  if len(z)>=8 and z['signal'].nunique()>1: vals.append(z['signal'].corr(z['f'],method='spearman'))
 q=np.array(vals); print(f'REGIME {a[:4]}-{b[:4]} dates={len(q)} IC={np.nanmean(q):.6f} ICIR={np.nanmean(q)/np.nanstd(q,ddof=1):.6f}')
