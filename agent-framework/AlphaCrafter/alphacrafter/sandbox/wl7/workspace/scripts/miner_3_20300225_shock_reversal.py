import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

D={}
for s in U:
 try: d=get_index_daily_data(s,days=4000)
 except Exception: d=None
 if d is None:
  try: d=get_stock_daily_data(s,days=4000)
  except Exception: d=None
 if d is not None: D[s]=d
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index()
# candidate: short-term shock reversal, normalized by own vol and gated by high cross-sectional dispersion (lagged)
r=np.log(px).diff(); vol=r.rolling(20,min_periods=15).std(); shock=-(r.rolling(5,min_periods=5).sum()/vol)
disp=r.rolling(5).std().mean(axis=1) # cross-asset average recent vol, observable t
# gate relative to 60d median, lag signal one day
sig=shock.shift(1)
rows=[]
for t in sig.index:
    fwd=np.log(px.shift(-10).loc[t]/px.loc[t])
    x=sig.loc[t]
    z=pd.concat([x,fwd],axis=1).dropna()
    if len(z)>=8:
        rows.append((t,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(out),'avg_n',out.n.mean(),'coverage',sig.notna().mean().mean(),'ic',out.ic.mean(),'icir',out.ic.mean()/out.ic.std(),'hit', (out.ic>0).mean())
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-02-25')]:
 q=out.loc[a:b].ic; print(a,b,'dates',len(q),'ic',q.mean(),'icir',q.mean()/q.std() if len(q)>1 else np.nan)
# turnover rank changes among valid dates
rank=sig.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna(); print('turnover',turn.mean())
for h in [1,5,10,20,40]:
 f=np.log(px.shift(-h)/px); rr=[]
 for t in sig.index:
  z=pd.concat([sig.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'ic',np.nanmean(rr),'dates',len(rr))
out.to_csv('scripts/miner_3_20300225_shock_reversal_dispersion_ic.csv')
sig.to_csv('scripts/miner_3_20300225_shock_reversal_dispersion_signal.csv')
