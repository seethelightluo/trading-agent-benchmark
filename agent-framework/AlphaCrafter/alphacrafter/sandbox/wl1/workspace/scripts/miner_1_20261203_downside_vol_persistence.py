import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2026-12-02'
def load(s):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return x.close.loc[:cut]
p=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=p.pct_change()
# Defensive downside-volatility persistence: prefer assets with low recent downside risk,
# but reward improving downside risk versus its medium-term baseline.
down=r.clip(upper=0)**2
dv20=np.sqrt(down.rolling(20,min_periods=15).mean())
dv60=np.sqrt(down.rolling(60,min_periods=40).mean())
f=(-(dv20/dv60)).shift(1)
rows=[]
for h in [5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(d)
 a=np.asarray(vals); print(f'{h}d dates={len(a)} avgN={np.mean(ns):.2f} IC={np.nanmean(a):.6f} ICIR={np.nanmean(a)/np.nanstd(a,ddof=1):.6f} hit={np.mean(a>0):.4f}')
 rows.extend([(d,h,v) for d,v in zip(ds,vals)])
rank=f.rank(axis=1,pct=True); print('coverage=%.6f turnover=%.6f'%(f.notna().sum(axis=1).div(15).mean(),rank.diff().abs().mean(axis=1).mean()))
for h in [10]:
 q=[(d,v) for d,hh,v in rows if hh==h]; s=pd.Series(dict(q));
 for y,g in s.groupby(s.index.year): print('year',y,'IC=%.6f n=%d'%(g.mean(),len(g)))
print('cutoff',cut)
