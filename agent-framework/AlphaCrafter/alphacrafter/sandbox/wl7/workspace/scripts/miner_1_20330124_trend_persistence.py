import numpy as np, pandas as pd, glob, os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={s:get_stock_daily_data(s,days=5000) for s in U}
close=pd.concat({s:d.set_index('date')['close'] for s,d in raw.items() if d is not None},axis=1).sort_index().ffill()
r=close.pct_change()
# Trend persistence: lagged 15-session return multiplied by fraction of positive sessions,
# then volatility-scaled. Three-session information buffer avoids lookahead.
ret15=close.shift(3)/close.shift(18)-1
persistence=r.gt(0).rolling(15,min_periods=12).mean().shift(3)
vol=r.rolling(40,min_periods=30).std().shift(3)
f=ret15*persistence/vol.replace(0,np.nan)
print('universe',close.shape[1],'dates',close.index.min(),close.index.max())
for h in [5,10,20]:
 fr=close.shift(-h)/close-1; vals=[]; dates=[]; ns=[]
 for dt in close.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 a=pd.Series(vals,index=pd.to_datetime(dates)).dropna()
 print(f'H{h} dates={len(a)} avg_n={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={np.mean(a>0):.4f}')
 print('thirds',[round(q.mean(),6) for q in np.array_split(a,3)])
f.rename_axis('date').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20330124_trend_persistence_signal.csv',index=False)
print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
print('last_signal_date',f.dropna(how='all').index.max())
# compare against all recoverable signal artifacts
cors=[]
for p in glob.glob('scripts/*_signal.csv'):
 try:
  x=pd.read_csv(p); x=x.pivot(index='date',columns='symbol',values='signal'); c=x.reindex(f.index).corrwith(f).abs().max()
  if pd.notna(c): cors.append((os.path.basename(p),float(c)))
 except Exception: pass
print('max_abs_library_correlation',max([x[1] for x in cors],default=None))
print('most correlated',sorted(cors,key=lambda x:x[1],reverse=True)[:5])
