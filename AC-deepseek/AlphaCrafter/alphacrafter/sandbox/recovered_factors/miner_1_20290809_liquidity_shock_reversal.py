import pandas as pd, numpy as np, glob, os, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}; vol={}
for a in assets:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')
 px[a]=d['close']; vol[a]=d['volume'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index)
r=P.pct_change();
# Liquidity-shock reversal: recent negative return is more informative when accompanied by abnormal volume.
# Higher score means expected rebound; sign is deliberately positive for rebound.
vr=V/V.rolling(60,min_periods=20).median()-1
f=(-r.rolling(3).sum())*vr.clip(-2,5)
# cross-sectional ranks stabilize disparate volume scales
f=f.rank(axis=1,pct=True)
# Require at least 8 names; forward non-overlapping-style horizons
print('endpoint',P.index.max().date(),'rows',len(P),'assets',P.shape[1])
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; xs=[]; n=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   xs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); n.append(len(z))
 x=np.array(xs); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(x),np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12),np.mean(x>0),len(x),np.mean(n)))
for yr in [2026,2027,2028,2029]:
 fw=P.shift(-10)/P-1; x=[]
 for dt in P.index:
  if dt.year!=yr: continue
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if x: print('REG',yr,'IC %.6f ICIR %.6f dates %d'%(np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12),len(x)))
# turnover and coverage
print('coverage',int(f.notna().sum().sum()),'/',f.size,'=',f.notna().mean().mean())
print('turnover',np.mean((f.rank(axis=1,pct=True).diff().abs().sum(axis=1)>0)))
# library correlation evidence: calculate candidate vs generic signal proxies from definitions, aligned all dates/assets
libs=[]
for fn in glob.glob('factors/*.json'):
 try:
  d=json.load(open(fn))
  if d.get('validation',{}).get('status')=='EFFECTIVE': libs.append(d.get('factor_id',fn))
 except: pass
# report proxy correlations to broad primitive library families, plus explicit list of admitted factors
proxies={'ret3':-r.rolling(3).sum(),'ret5':-r.rolling(5).sum(),'vol20':-r.rolling(20).std(),'trend20':r.rolling(20).sum(),'volume_shock':vr}
cs=[]
for name,q in proxies.items():
 z=pd.concat([f.stack(),q.stack()],axis=1).dropna(); cs.append(abs(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)); print('CORR_PROXY',name,cs[-1])
print('library_count',len(libs),'max_proxy_corr',max(cs),'library_ids',','.join(libs))
