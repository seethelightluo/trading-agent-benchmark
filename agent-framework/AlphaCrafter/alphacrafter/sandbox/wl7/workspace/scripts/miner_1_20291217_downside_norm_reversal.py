import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
TODAY='2029-12-17'; HORIZONS=[5,10,20]; U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
try: U=get_account_dict().get('watch_list',[]) or U
except Exception: pass
px={}
for s in U:
    try: d=get_index_daily_data(s,days=4000)
    except Exception: d=None
    if d is None or len(d)<150:
        try: d=get_stock_daily_data(s,days=4000)
        except Exception: d=None
    if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]
for h in HORIZONS:
 rows=[]
 for i in range(80,len(P)-h):
  vals=[]
  for s in U:
   if s not in P: continue
   x=P[s].iloc[:i+1].dropna()
   if len(x)<61 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]): continue
   r=x.pct_change().dropna(); recent=r.iloc[-10:-1]; down=recent[recent<0]
   dv=down.std() if len(down)>=3 else np.nan
   # lagged 10-day loss, scaled by downside dispersion; stronger only for selloffs
   r10=x.iloc[-1]/x.iloc[-11]-1
   sig=-r10/(dv*np.sqrt(10)) if np.isfinite(dv) and dv>0 else np.nan
   if np.isfinite(sig): vals.append((s,sig,P[s].iloc[i+h]/x.iloc[-1]-1))
  if len(vals)>=8:
   rows.extend([(P.index[i],)+z for z in vals])
 df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna()
 ic=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna()
 ranks=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
 turn=ranks.diff().abs().mean(axis=1).dropna().mean()
 print('H',h,'dates',len(ic),'avg_names',df.groupby('date').size().mean(),'coverage',df.symbol.nunique()/len(U),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turn',turn)
 if len(ic):
  for j,(a,b) in enumerate([(0,len(ic)//3),(len(ic)//3,2*len(ic)//3),(2*len(ic)//3,len(ic))]):
   q=ic.iloc[a:b]; print(' regime',j,len(q),q.mean(),q.mean()/q.std())
 if h==20: df.to_csv('scripts/miner_1_20291217_downside_norm_reversal_signal.csv',index=False)
