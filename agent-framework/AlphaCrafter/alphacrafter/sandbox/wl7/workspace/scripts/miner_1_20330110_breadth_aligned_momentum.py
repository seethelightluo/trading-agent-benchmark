import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={s:get_stock_daily_data(s,days=5000) for s in U}
close=pd.concat({s:d.set_index('date')['close'] for s,d in raw.items() if d is not None},axis=1).sort_index().ffill()
r=close.pct_change()
# Regime-adaptive medium trend: rank-normalized 20d momentum, with its direction
# aligned to the contemporaneous cross-asset breadth regime, all inputs lagged 3 sessions.
mom=close.shift(3)/close.shift(23)-1
breadth=(r.rolling(20).mean().gt(0).mean(axis=1)-0.5).shift(3)
f=mom.mul(np.sign(breadth).replace(0,np.nan),axis=0)
print('universe',close.shape[1],'dates',close.index.min(),close.index.max())
for h in [10,20]:
 fr=close.shift(-h)/close-1; vals=[]; dates=[]; ns=[]
 for dt in close.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 a=pd.Series(vals,index=pd.to_datetime(dates)).dropna()
 print(f'H{h} dates={len(a)} avg_n={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={np.mean(a>0):.4f}')
 print('thirds',[round(q.mean(),6) for q in np.array_split(a,3)])
f.rename_axis('date').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20330110_breadth_aligned_momentum_signal.csv',index=False)
print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
print('last_signal_date',f.dropna(how='all').index.max())
# regime counts
print('breadth positive',float((breadth>0).mean()),'negative',float((breadth<0).mean()))
# library artifact correlations where available
import glob, os
cors=[]
for p in glob.glob('scripts/*_signal.csv'):
 try:
  x=pd.read_csv(p); x=x.pivot(index='date',columns='symbol',values='signal'); c=x.reindex(f.index).corrwith(f).abs().max()
  if pd.notna(c): cors.append((os.path.basename(p),float(c)))
 except Exception: pass
print('max_abs_library_correlation',max([x[1] for x in cors],default=None))
print('most correlated',sorted(cors,key=lambda x:x[1],reverse=True)[:3])
