import pandas as pd, numpy as np, glob, json, os
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-09-09')
def get(s):
 for p in ['../persistent/stock_data/'+s+'.csv','../persistent/index_data/'+s+'.csv']:
  if os.path.exists(p):
   x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); return x[x.index<=CUT]
 raise FileNotFoundError(s)
raw={s:get(s) for s in ASSETS}; C=pd.concat([x.close.rename(s) for s,x in raw.items()],axis=1); ret=C.pct_change()
v=get('VIX').close.pct_change().rename('vix')
# Defensive VIX-beta: negative rolling correlation with contemporaneous VIX changes.
# Higher value = lower/risk-offsetting VIX sensitivity, deliberately a macro conditional risk measure.
F=pd.concat([ret[s].rolling(20,min_periods=15).corr(v).mul(-1).rename(s) for s in ASSETS],axis=1)
def library():
 out={}
 for path in glob.glob('factors/*.json'):
  if path.endswith('.bak'): continue
  fid=json.load(open(path))['factor_id']
  if 'relative_volume' in fid: out[fid]=pd.concat([np.log(raw[s].volume/raw[s].volume.rolling(20,min_periods=15).mean()).rename(s) for s in ASSETS],axis=1)
  elif 'realized_volatility' in fid: out[fid]=ret.rolling(20,min_periods=15).std()
  elif 'volscaled_reversal_1obs' in fid: out[fid]=-ret/ret.rolling(20,min_periods=15).std()
  elif 'volnorm_reversal' in fid: out[fid]=-C.pct_change(5)/ret.rolling(5,min_periods=4).std()
  else: out[fid]=C.pct_change(20)/ret.rolling(20,min_periods=15).std()
 return out
def met(h, subset=None):
 f=F if subset is None else F.loc[subset[0]:subset[1]]; R=(C.shift(-h)/C-1).reindex(f.index); a=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8: a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(a); return len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean()),float(np.mean(ns))
print('FACTOR=vix_defensive_beta_20obs cutoff',CUT.date()); print('cells',int(F.notna().sum().sum()),'/',F.size,'coverage',F.notna().mean().mean())
for h in [1,5,10,20]: print('H',h,'n IC ICIR hit nassets',met(h))
for n,lo,hi in [('2020','2020-01-01','2020-12-31'),('2021-22','2021-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-09-09')]: print('REGIME10',n,met(10,(lo,hi)))
r=F.rank(axis=1,pct=True); print('rank_turnover',r.diff().abs().stack().mean())
mx=0
for n,L in library().items():
 z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna(); rho=spearmanr(z.a,z.b).statistic; mx=max(mx,abs(rho)); print('LIBCORR',n,len(z),rho)
print('max_abs_library_correlation',mx)
