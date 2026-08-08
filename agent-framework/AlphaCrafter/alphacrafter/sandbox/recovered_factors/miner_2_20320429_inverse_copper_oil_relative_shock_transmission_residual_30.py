import os, glob, json
import pandas as pd, numpy as np
from scipy.stats import spearmanr

# One idea: copper-oil relative-shock transmission.  A rising copper/oil ratio
# signals growth-vs-energy disagreement; rank assets by their *lagged* 30d beta
# to the shock, remove generic trend/volatility, and test the inverse residual.
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-04-28') # previous completed session to current 2032-04-29

def load(path, col='close'):
 d=pd.read_csv(path,parse_dates=['date']).set_index('date')[col].astype(float)
 return d[d.index<=END]
px=pd.concat({a:load('../persistent/stock_data/'+a+'.csv') for a in ASSETS},axis=1)
cu=load('../persistent/stock_data/COPPER.csv'); oil=load('../persistent/stock_data/WTI.csv')
# inner calendar only prevents using a non-existent asset close
idx=px.index.union(cu.index).union(oil.index).sort_values()
px=px.reindex(idx).ffill(); cu=cu.reindex(idx).ffill(); oil=oil.reindex(idx).ffill()
r=np.log(px/px.shift(1)); shock=(np.log(cu/oil).diff(5)).shift(1) # fully lagged
# rolling beta to relative macro shock; centered product / shock variance
beta=r.rolling(30,min_periods=22).cov(shock).div(shock.rolling(30,min_periods=22).var(),axis=0).shift(1)
vol=r.rolling(20,min_periods=15).std().shift(1); trend=r.rolling(20,min_periods=15).sum().shift(1)
# daily cross-sectional residual beta against own trend and volatility, then inverse direction
fac=pd.DataFrame(index=idx,columns=ASSETS,dtype=float)
for d in idx:
 y=beta.loc[d]; x=pd.concat([trend.loc[d],vol.loc[d]],axis=1); ok=y.notna()&x.notna().all(axis=1)
 if ok.sum()>=8:
  X=np.c_[np.ones(ok.sum()),x.loc[ok].values]
  fac.loc[d,ok]=- (y.loc[ok].values-X@np.linalg.lstsq(X,y.loc[ok].values,rcond=None)[0])

def report(h):
 fwd=np.log(px.shift(-h)/px)
 ics=[]; ns=[]
 for d in idx:
  z=pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(ics); mean=a.mean(); sd=a.std(ddof=1); ir=mean/sd if sd else np.nan
 print(f'H={h} dates={len(a)} mean_n={np.mean(ns):.2f} IC={mean:.6f} ICIR={ir:.6f} hit={(a>0).mean():.4f} se={sd/np.sqrt(len(a)):.6f}')
 return a
print('candidate=inverse_copper_oil_relative_shock_transmission_residual_30')
print('visible_endpoint',END.date(),'cells',int(fac.notna().sum().sum()),'coverage',fac.notna().mean().mean())
for h in (1,5,10,20): report(h)
# turnover, and 20d broad date-regimes
rank=fac.rank(axis=1,pct=True); print('mean_rank_turnover',rank.diff().abs().stack().mean())
fwd=np.log(px.shift(-20)/px); rows=[]
for d in idx:
 z=pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
z=pd.DataFrame(rows,columns=['date','ic'])
for name, q in [('2026_2027',z.date<'2028-01-01'),('2028_current',z.date>='2028-01-01')]:
 a=z.loc[q,'ic']; print('REGIME',name,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
# Print active library metadata; a passing candidate requires reconstructible correlations.
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append((os.path.basename(p),j.get('factor_id')))
 except: pass
print('active_library',len(active),active)
print('LIBRARY_CORRELATION=NOT_COMPUTED: definitions are heterogeneous; do not admit without reconstructed signals')
