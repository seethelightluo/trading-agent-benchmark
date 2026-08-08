"""miner_1: test trailing downside-return participation, a drawdown-shape factor."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-08-12')
prices={}
for a in ASSETS:
 p=(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').close.astype(float).sort_index())
 prices[a]=p
# Fraction of trailing absolute return variation accounted for by down days. Higher values indicate a downside-dominated path, separate from return magnitude.
def down_part(p):
 r=p.pct_change(fill_method=None); return r.clip(upper=0).abs().rolling(20,min_periods=15).sum()/r.abs().rolling(20,min_periods=15).sum()
factor=pd.DataFrame({a:down_part(p) for a,p in prices.items()}).sort_index()
def evaluate(h):
 future=pd.DataFrame({a:p.shift(-h)/p-1 for a,p in prices.items()}).reindex(factor.index); obs=[];ns=[]
 for d in factor.index:
  z=pd.concat([factor.loc[d],future.loc[d]],axis=1).dropna()
  if len(z)>=8: obs.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic));ns.append(len(z))
 ic=pd.Series(dict(obs));mean=ic.mean();sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(mean),'daily_paper_icir':float(mean/sd),'ic_std':float(sd),'ic_hit_ratio':float((ic>0).mean()),'n_dates':len(ic),'mean_valid_instruments':float(np.mean(ns)),'yearly_ic':{str(y):float(v.mean()) for y,v in ic.groupby(ic.index.year)}}
metrics={}
for h in [1,5,10,20]:
 ic,m=evaluate(h);metrics[h]=m;print('HORIZON',h,json.dumps(m,sort_keys=True))
ic=evaluate(10)[0]; market=pd.DataFrame({a:p.pct_change(fill_method=None) for a,p in prices.items()}).mean(axis=1).rolling(20,min_periods=15).sum().reindex(ic.index); med=market.median()
for label,mask in [('higher_market',market>=med),('lower_market',market<med)]:
 x=ic[mask];print('REGIME',label,'n_dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)))
t=[]
for i in range(1,len(factor)):
 z=pd.concat([factor.iloc[i-1],factor.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
mapping={'miner_1_ravmom_20obs':'scripts/miner_1_20260716_ravmom20_signal.pkl','miner_1_volnorm_reversal_5obs':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_2_realized_volatility_20obs':'scripts/miner_2_20260716_realized_volatility20_signal.pkl','miner_3_relative_volume_participation_20d':'scripts/miner_3_20260716_relative_volume_participation_20d_signal.pkl','miner_3_risk_adjusted_trend_20d':'scripts/miner_3_20260716_risk_adjusted_trend_20d_signal.pkl'}
corrs=[]
for fp in glob.glob('factors/*.json'):
 meta=json.load(open(fp))
 if meta.get('validation',{}).get('status')!='EFFECTIVE':continue
 name=meta['factor_id'];path=mapping.get(name)
 if not path or not os.path.exists(path): print('LIBRARY_CORR',name,'MISSING');corrs.append(np.nan);continue
 lib=pd.read_pickle(path).reindex(index=factor.index,columns=ASSETS);z=pd.concat([factor.stack().rename('new'),lib.stack().rename('lib')],axis=1).dropna();rho=spearmanr(z.new,z.lib).statistic if len(z)>=8 else np.nan
 print('LIBRARY_CORR',name,'n_pairs',len(z),'spearman',rho);corrs.append(abs(rho))
print('FACTOR downside_participation_20obs');print('PERIOD',factor.index.min().date(),END.date(),'coverage',float(factor.notna().mean().mean()),'panel_dates',len(factor),'mean_names',float(factor.notna().sum(axis=1).mean()),'mean_rank_turnover_1d',float(np.mean(t)))
print('DECAY',json.dumps({str(h):{'ic':metrics[h]['daily_paper_ic'],'icir':metrics[h]['daily_paper_icir'],'n_dates':metrics[h]['n_dates']} for h in metrics}))
print('MAX_ABS_LIBRARY_CORRELATION',max(corrs) if corrs and all(np.isfinite(corrs)) else 'MISSING')
factor.to_pickle('scripts/miner_1_20260813_downside_participation20_signal.pkl')
