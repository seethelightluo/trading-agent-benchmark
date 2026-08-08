"""miner_2 one-idea test: low market-beta defensive factor, evaluated only with history through 2026-08-12."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-08-12')
close={}; volume={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 close[a]=d.close.astype(float); volume[a]=d.volume.astype(float)
p=pd.DataFrame(close).sort_index(); r=p.pct_change(fill_method=None)
# Equal-weight observed market return; score is negative own 20-observation beta. High score means defensive / low systematic sensitivity.
market=r.mean(axis=1,skipna=True)
f=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(market)/market.rolling(20,min_periods=15).var() for a in ASSETS})
# Build all five admitted signals directly from their persisted definitions (m1 and m3 trend definitions are identical).
trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
lib={
 'miner_1_ravmom_20obs':trend,
 'miner_3_risk_adjusted_trend_20d':trend,
 'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),
 'miner_2_realized_volatility_20obs':-r.rolling(20,min_periods=15).std(),
 'miner_3_relative_volume_participation_20d':np.log(pd.DataFrame(volume)/pd.DataFrame(volume).rolling(20,min_periods=15).mean())
}
def calc(h):
 fw=p.shift(-h)/p-1; obs=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d].rename('x'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8: obs.append((d,spearmanr(z.x,z.y).statistic));ns.append(len(z))
 s=pd.Series(dict(obs)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_std':float(sd),'ic_hit_ratio':float((s>0).mean()),'n_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
metrics={}
for h in [1,5,10,20]:
 s,m=calc(h);metrics[h]=m;print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for name,mask in [('2020',s.index<'2021-01-01'),('2021_22',(s.index>='2021-01-01')&(s.index<'2023-01-01')),('2023_24',(s.index>='2023-01-01')&(s.index<'2025-01-01')),('2025_26',s.index>='2025-01-01')]:
   q=s[mask];print('REGIME',name,'n',len(q),'IC',float(q.mean()),'ICIR',float(q.mean()/q.std(ddof=1)))
# rank turnover and correlation evidence are pooled date-instrument observations, using exact signal definitions.
turn=[]
for i in range(1,len(f)):
 z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
corr={}
for name,x in lib.items():
 z=pd.concat([f.stack().rename('new'),x.stack().rename('old')],axis=1).dropna()
 corr[name]=float(spearmanr(z.new,z.old).statistic) if len(z)>=8 else np.nan
 print('LIBRARY_CORR',name,'n_pairs',len(z),'rho',corr[name])
print('FACTOR low_market_beta_20obs')
print('PERIOD',f.index.min().date(),END.date(),'panel_dates',len(f),'coverage',float(f.notna().mean().mean()),'mean_names',float(f.notna().sum(axis=1).mean()),'mean_rank_turnover_1d',float(np.mean(turn)))
print('DECAY',json.dumps({str(h):{'ic':metrics[h]['daily_paper_ic'],'icir':metrics[h]['daily_paper_icir'],'n_dates':metrics[h]['n_dates']} for h in metrics}))
print('MAX_ABS_LIBRARY_CORRELATION',float(max(abs(x) for x in corr.values())))
