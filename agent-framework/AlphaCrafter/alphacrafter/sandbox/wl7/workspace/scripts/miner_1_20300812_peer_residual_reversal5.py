import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try: d=fn(s,days=4000)
        except Exception: pass
        if d is not None and len(d)>200: break
    if d is not None:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Idiosyncratic 5-day return versus contemporaneous equal-weight benchmark,
# divided by asset-specific residual 20d volatility; lag one day to avoid lookahead.
bench=r.mean(axis=1)
resid=r.sub(bench,axis=0)
f=-(resid.rolling(5).sum()/resid.rolling(20).std().replace(0,np.nan)).shift(1)
rows=[]
for dt in f.index:
    y=p.pct_change(10).shift(-10).loc[dt]
    z=pd.concat([f.loc[dt],y],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=np.array([x[1] for x in rows]); cut=len(a)//3
print('factor=lagged_peer_residual_reversal5 dates',len(a),'avg_n',round(np.mean([x[2] for x in rows]),2),'assets',len(U))
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),np.mean([x[2] for x in rows])/len(U)))
print('regimes',[(round(float(np.nanmean(q)),6),round(float(np.nanmean(q)/np.nanstd(q,ddof=1)),6)) for q in (a[:cut],a[cut:2*cut],a[2*cut:])])
for h in [1,5,10,20,40]:
 y=p.pct_change(h).shift(-h); aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(float(np.nanmean(aa)),6),len(aa))
q=f.rank(axis=1,pct=True); print('turnover',float(q.diff().abs().mean(axis=1).mean()))
out=f.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20300812_peer_residual_reversal5_signal.csv')
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20300812_peer_residual_reversal5_ic.csv',index=False)
