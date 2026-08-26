import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    x=get_stock_daily_data(s, days=3000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.drop_duplicates('date').set_index('date').sort_index()
        raw[s]=x['close'].astype(float)
p=pd.DataFrame(raw).sort_index()
# common cross-section, aligned prices; factor is residual reversal versus cross-sectional median return
r20=p.pct_change(20); vol40=p.pct_change().rolling(40).std()*np.sqrt(252)
rows=[]
for i,d in enumerate(p.index):
    if i+20>=len(p): break
    vals=r20.loc[d]; vv=vol40.loc[d]
    valid=vals.notna() & vv.notna() & (vv>1e-8)
    if valid.sum()<8: continue
    med=vals[valid].median()
    # negative residual: contrarian to asset return relative to contemporaneous universe
    f=-(vals-med)/vv
    f=f[valid].replace([np.inf,-np.inf],np.nan).dropna()
    if len(f)<8: continue
    fut=p.iloc[i+1:i+21].iloc[-1]/p.iloc[i+1]-1
    fut=fut.reindex(f.index).dropna(); common=f.index.intersection(fut.index)
    if len(common)>=8:
        rows.append((d, f.reindex(common), fut.reindex(common)))
ics=[]; signals=[]
for d,f,y in rows:
    ic=f.corr(y, method='spearman')
    if np.isfinite(ic):
        ics.append(ic); signals.append((d,f))
ics=np.array(ics)
print('dates',len(ics),'avg_n',np.mean([len(f) for _,f,_ in rows]),'instruments',len(p.columns))
print('IC20 %.6f ICIR %.6f hit %.4f' % (ics.mean(), ics.mean()/ics.std(ddof=1), (ics>0).mean()))
# regimes with enough history
for a,b in [('2026-01-01','2028-12-31'),('2029-01-01','2031-04-17')]:
 z=np.array([q for (d,_,_),q in zip(rows,ics) if pd.Timestamp(a)<=d<=pd.Timestamp(b)])
 print(a,b,'n',len(z),'ic', (z.mean() if len(z) else np.nan),'hit',(z>0).mean() if len(z) else np.nan)
# rank turnover proxy across adjacent observations
turn=[]
for (_,f),(_,g) in zip(signals[:-1],signals[1:]):
 c=f.index.intersection(g.index)
 if len(c)>=8: turn.append(np.mean(f[c].rank(pct=True).sub(g[c].rank(pct=True)).abs()))
print('coverage %.6f turnover %.6f' % (np.mean([len(f)/15 for _,f,_ in rows]),np.mean(turn)))
# decay diagnostics
for h in [5,10,40]:
 z=[]
 for i,d in enumerate(p.index):
  if i+ h>=len(p): continue
  vals=r20.loc[d]; vv=vol40.loc[d]; valid=vals.notna()&vv.notna()&(vv>1e-8)
  if valid.sum()<8: continue
  f=(-(vals[valid]-vals[valid].median())/vv[valid]).replace([np.inf,-np.inf],np.nan).dropna()
  y=(p.iloc[i+h]/p.iloc[i+1]-1).reindex(f.index).dropna(); c=f.index.intersection(y.index)
  if len(c)>=8: z.append(f[c].corr(y[c],method='spearman'))
 print('h',h,'n',len(z),'ic',np.nanmean(z),'icir',np.nanmean(z)/np.nanstd(z,ddof=1))
