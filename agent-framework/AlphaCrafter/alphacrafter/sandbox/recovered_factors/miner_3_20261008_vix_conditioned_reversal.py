import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
from pathlib import Path

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
prices={}
for a in assets:
 p=base/(a+'.csv')
 if p.exists(): prices[a]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close']
px=pd.DataFrame(prices).sort_index(); common=px.index.intersection(vix.index); px=px.loc[common]; vv=vix.loc[common]
# VIX-conditioned short-horizon reversal: recent asset loss is more likely to mean-revert
# when volatility is elevated relative to its trailing 60-day median; all inputs lagged one day.
r1=px.pct_change(1); vratio=vv/vv.rolling(60,min_periods=30).median()
sig=-r1.mul(vratio,axis=0)
# forward returns from t to t+h, with signal at t using close t (decision after t)
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1
 ics=[]; n=[]; dates=[]
 for d in sig.index:
  x=sig.loc[d]; y=fwd.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   ics.append(spearmanr(x[ok],y[ok]).statistic); n.append(ok.sum()); dates.append(d)
 z=pd.Series(ics,index=dates)
 print('H',h,'dates',len(z),'mean_inst',round(np.mean(n),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 print('years',z.groupby(z.index.year).mean().round(5).to_dict())
# turnover and coverage
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('coverage',round(sig.notna().mean().mean(),4),'turnover',round(turn,4),'cells',int(sig.notna().sum().sum()))
# library pooled correlations from factor signal files, recompute known expressions where possible
libs={}
for f in glob.glob('factors/*.json'):
 if '.bak' in f: continue
 import json
 j=json.load(open(f)); fid=j['factor_id']; expr=j.get('calculation',{}).get('expression','')
 # use stored expression heuristics
 if 'volume' in expr.lower(): continue
 if 'reversal' in fid:
  s=-px.pct_change(5)/px.pct_change().rolling(20).std()
 elif 'volatility' in fid:
  s=-px.pct_change().rolling(20).std()
 elif 'trend' in fid or 'ravmom' in fid:
  s=px.pct_change(20)/px.pct_change().rolling(20).std()
 else: continue
 a,b=sig.align(s,join='inner'); ok=a.notna()&b.notna(); libs[fid]=spearmanr(a[ok],b[ok]).statistic
print('library_corr', {k:round(v,5) for k,v in libs.items()})
print('max_abs',round(max([abs(v) for v in libs.values()] or [0]),6))
