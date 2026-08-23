import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym):
 d=get_stock_daily_data(sym, days=5000)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
px=pd.concat({s:load(s) for s in U},axis=1).sort_index(); ret=px.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); vc='close' if 'close' in vix else vix.columns[-1]
vix=vix.set_index('date')[vc].astype(float).reindex(px.index).ffill()
stress=(vix.pct_change(5)>0.04).astype(float); r20=px.pct_change(20); vol20=ret.rolling(20).std()*np.sqrt(252)
base=r20.sub(r20.median(axis=1),axis=0).div(vol20).replace([np.inf,-np.inf],np.nan)
f=base*(1.0+0.45*stress.values[:,None]); med60=r20.rolling(60).median().T.median().T
f=f.where(r20.ge(med60,axis=0), f*0.65)
f.to_csv('scripts/miner_2_20300221_macro_stress_relative_strength_signal.csv',index_label='date')
for h in [5,10,20]:
 fr=px.shift(-h).div(px)-1; ics=[]; cov=[]; ns=[]
 for dt in px.index:
  a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   q=a[ok].corr(b[ok],method='spearman')
   if pd.notna(q): ics.append(q); cov.append(ok.mean()); ns.append(ok.sum())
 z=pd.Series(ics); ic=z.mean(); ir=ic/z.std(ddof=1)*np.sqrt(252)
 print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={ic:.8f} ICIR={ir:.8f} hit={(z>0).mean():.4f} coverage={np.mean(cov):.4f}')
ranks=f.rank(axis=1,pct=True); print('rows',len(px),'instruments',px.shape[1],'turnover',float((ranks-ranks.shift()).abs().mean(axis=1).mean()),'last_signal',px.index[-1].date())
