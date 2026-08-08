import pandas as pd,numpy as np,json,glob
from scipy.stats import spearmanr
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); px=pd.DataFrame({a:pd.read_csv(base/(a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index()
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill()
r=px.pct_change(); dm=dxy.pct_change(20)
# Macro-conditioned reversal: fade yesterday's return, with sign scaled by dollar trend.
f=-r.mul(np.sign(dm),axis=0)
print('FACTOR dxy_conditioned_reversal dates',len(f),'assets',len(A))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; q=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 q=pd.Series(q,index=ds); print('H',h,'dates',len(q),'mean_inst',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'years',q.groupby(q.index.year).mean().round(5).to_dict())
rank=f.rank(axis=1,pct=True); print('coverage',round(f.notna().stack().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
# correlations against admitted library proxy signals and exact persisted expression approximations
proxies={}
for jfile in glob.glob('factors/*.json'):
 if '.bak' in jfile: continue
 j=json.load(open(jfile)); fid=j['factor_id']; ex=j.get('calculation',{}).get('expression','').lower()
 if 'volume' in ex: continue
 if 'reversal' in fid: s=-r.rolling(5).sum()/r.rolling(20).std()
 elif 'volatility' in fid: s=-r.rolling(20).std()
 elif 'trend' in fid or 'ravmom' in fid: s=r.rolling(20).sum()/r.rolling(20).std()
 elif 'peer' in fid: s=r.rolling(2).sum().sub(r.mean(axis=1),axis=0)
 else: continue
 aa,bb=f.align(s,join='inner'); ok=aa.notna()&bb.notna(); proxies[fid]=float(spearmanr(aa.to_numpy()[ok.to_numpy()],bb.to_numpy()[ok.to_numpy()]).statistic)
print('library_corr', {k:round(v,5) for k,v in proxies.items()}); print('max_abs',round(max([abs(x) for x in proxies.values()] or [0]),6))
