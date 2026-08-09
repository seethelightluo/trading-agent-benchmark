"""miner_2: test 20-observation return skewness as an independent cross-asset factor."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-07-15')
prices={}
for a in ASSETS:
 p=(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END')
    .drop_duplicates('date').set_index('date').close.astype(float).sort_index())
 prices[a]=p
# Standardized third central moment of daily returns, trailing only; positive tail asymmetry gets high rank.
factor=pd.DataFrame({a: p.pct_change(fill_method=None).rolling(20,min_periods=15).skew() for a,p in prices.items()}).sort_index()
def evaluate(h):
 future=pd.DataFrame({a:p.shift(-h)/p-1 for a,p in prices.items()}).reindex(factor.index)
 obs=[]; ns=[]
 for d in factor.index:
  z=pd.concat([factor.loc[d],future.loc[d]],axis=1).dropna()
  if len(z)>=8: obs.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic));ns.append(len(z))
 ic=pd.Series(dict(obs)); mean=ic.mean(); sd=ic.std(ddof=1)
 return ic,dict(daily_paper_ic=float(mean),daily_paper_icir=float(mean/sd),ic_std=float(sd),ic_hit_ratio=float((ic>0).mean()),n_dates=len(ic),mean_valid_instruments=float(np.mean(ns)),yearly_ic={str(y):float(s.mean()) for y,s in ic.groupby(ic.index.year)})
metrics={}
for h in [1,5,10,20]:
 ic,m=evaluate(h);metrics[h]=m;print('HORIZON',h,json.dumps(m,sort_keys=True))
ic=evaluate(10)[0]
ret=pd.DataFrame({a:p.pct_change(fill_method=None) for a,p in prices.items()})
reg=ret.mean(axis=1).rolling(20,min_periods=15).sum().reindex(ic.index); med=reg.median()
for label,mask in [('higher_market',reg>=med),('lower_market',reg<med)]:
 x=ic[mask];print('REGIME',label,'n_dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)))
# Date-to-date rank movement.
t=[]
for i in range(1,len(factor)):
 z=pd.concat([factor.iloc[i-1],factor.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
# Evidence against only factor files that are actually effective.
corrs=[]
for fp in glob.glob('factors/*.json'):
 try: meta=json.load(open(fp))
 except: continue
 if meta.get('validation',{}).get('status')!='EFFECTIVE':continue
 name=meta['factor_id']; candidates=glob.glob('scripts/*_signal.pkl')
 matches=[x for x in candidates if name.replace('miner_','miner_').split('_',1)[1].split('obs')[0].replace('_','') in os.path.basename(x).replace('_','')]
 # explicit mapping avoids accidental candidate inclusion
 mapping={'miner_1_ravmom_20obs':'scripts/miner_1_20260716_ravmom20_signal.pkl','miner_1_volnorm_reversal_5obs':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_2_realized_volatility_20obs':'scripts/miner_2_20260716_realized_volatility20_signal.pkl','miner_3_relative_volume_participation_20d':'scripts/miner_3_20260716_relative_volume_participation_20d_signal.pkl','miner_3_risk_adjusted_trend_20d':'scripts/miner_3_20260716_risk_adjusted_trend_20d_signal.pkl'}
 path=mapping.get(name)
 if not path or not os.path.exists(path): print('LIBRARY_CORR',name,'MISSING');corrs.append(np.nan);continue
 lib=pd.read_pickle(path).reindex(index=factor.index,columns=ASSETS)
 z=pd.concat([factor.stack().rename('new'),lib.stack().rename('lib')],axis=1).dropna();rho=spearmanr(z.new,z.lib).statistic if len(z)>=8 else np.nan
 print('LIBRARY_CORR',name,'n_pairs',len(z),'spearman',rho);corrs.append(abs(rho))
print('FACTOR return_skewness_20obs')
print('PERIOD',factor.index.min().date(),END.date(),'coverage',float(factor.notna().mean().mean()),'panel_dates',len(factor),'mean_names',float(factor.notna().sum(axis=1).mean()),'mean_rank_turnover_1d',float(np.mean(t)))
print('DECAY',json.dumps({str(h):{'ic':metrics[h]['daily_paper_ic'],'icir':metrics[h]['daily_paper_icir'],'n_dates':metrics[h]['n_dates']} for h in metrics}))
print('MAX_ABS_LIBRARY_CORRELATION',max(corrs) if corrs and all(np.isfinite(corrs)) else 'MISSING')
factor.to_pickle('scripts/miner_2_20260730_return_skewness20_signal.pkl')
