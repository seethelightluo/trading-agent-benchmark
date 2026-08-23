import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change();
for mode in ['conflict_only','trend_gate']:
 rows=[]
 for t in p.index:
  if len(r.loc[:t])<25: continue
  rr=p.loc[t]/p.shift(5).loc[t]-1; tr=p.loc[t]/p.shift(20).loc[t]-1; down=r.loc[:t].tail(20).clip(upper=0).std()
  f=-rr/(down+1e-8); agree=np.sign(rr*tr).fillna(0)
  f=f*(1-agree) if mode=='conflict_only' else f*(1-0.75*agree); f=f.replace([np.inf,-np.inf],np.nan)
  for s in U:
   if s in f and pd.notna(f[s]): rows.append((t,s,f[s]))
 F=pd.DataFrame(rows,columns=['date','symbol','factor']).set_index(['date','symbol']); fr=p.pct_change(10).shift(-10); a=[]; cov=[]
 for t,g in F.groupby(level=0):
  x=g.factor.droplevel(0); y=fr.loc[t].reindex(x.index); z=pd.concat([x,y.rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(z.factor.corr(z.y,method='spearman'));cov.append(len(z)/15)
 a=np.array(a); print(mode,len(a),np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0),np.mean(cov))
 F.reset_index().to_csv('scripts/miner_2_20290111_'+mode+'_signal.csv',index=False)
