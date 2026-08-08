"""miner_1: validate one factor -- 20-observation trend consistency (positive-day share). API data respects cursor."""
import glob,json,os
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']
prices={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 prices[a]=d.drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
# Higher scores mean a larger fraction of the last 20 completed sessions rose: persistent breadth, distinct from return magnitude.
factor=pd.DataFrame({a:(p.pct_change(fill_method=None)>0).astype(float).rolling(20,min_periods=15).mean().where(p.pct_change(fill_method=None).notna().rolling(20,min_periods=15).sum()>=15) for a,p in prices.items()}).sort_index()
def calc(h):
 fw=pd.DataFrame({a:p.shift(-h)/p-1 for a,p in prices.items()}).reindex(factor.index)
 obs=[]; ns=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8: obs.append((dt,spearmanr(z.f,z.r).statistic));ns.append(len(z))
 ic=pd.Series(dict(obs)); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_std':float(sd),'ic_hit_ratio':float((ic>0).mean()),'n_dates':len(ic),'mean_valid_instruments':float(np.mean(ns))}
allm={}
for h in [1,5,10,20]:
 ic,m=calc(h);allm[h]=m;print('HORIZON',h,json.dumps(m))
 for n,mask in [('2020',ic.index<'2021-01-01'),('2021_22',(ic.index>='2021-01-01')&(ic.index<'2023-01-01')),('2023_24',(ic.index>='2023-01-01')&(ic.index<'2025-01-01')),('2025_26',ic.index>='2025-01-01')]:
  x=ic[mask];print(' REGIME',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
turn=[]
for i in range(1,len(factor)):
 z=pd.concat([factor.iloc[i-1],factor.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
mapping={'miner_1_ravmom_20obs':'scripts/miner_1_20260716_ravmom20_signal.pkl','miner_1_volnorm_reversal_5obs':'scripts/miner_1_20260716_volnorm_reversal5_signal.pkl','miner_2_realized_volatility_20obs':'scripts/miner_2_20260716_realized_volatility20_signal.pkl','miner_3_relative_volume_participation_20d':'scripts/miner_3_20260716_relative_volume_participation_20d_signal.pkl','miner_3_risk_adjusted_trend_20d':'scripts/miner_3_20260716_risk_adjusted_trend_20d_signal.pkl'}
corr=[]
for fp in glob.glob('factors/*.json'):
 try: meta=json.load(open(fp))
 except: continue
 if meta.get('validation',{}).get('status')!='EFFECTIVE':continue
 name=meta['factor_id']; path=mapping.get(name)
 if not path or not os.path.exists(path): print('LIBRARY_CORR',name,'MISSING');corr.append(np.nan);continue
 lib=pd.read_pickle(path).reindex(index=factor.index,columns=assets)
 z=pd.concat([factor.stack().rename('new'),lib.stack().rename('old')],axis=1).dropna()
 rho=spearmanr(z.new,z.old).statistic if len(z)>=8 else np.nan
 print('LIBRARY_CORR',name,'pairs',len(z),'rho',rho);corr.append(abs(rho))
print('FACTOR trend_consistency_20obs = mean(return>0,20)')
print('PERIOD',factor.index.min().date(),factor.index.max().date(),'instruments',len(assets),'signal_coverage',float(factor.notna().mean().mean()),'mean_rank_turnover',float(np.mean(turn)))
print('DECAY',json.dumps({str(h):{'ic':allm[h]['daily_paper_ic'],'icir':allm[h]['daily_paper_icir'],'dates':allm[h]['n_dates']} for h in allm}))
print('MAX_ABS_LIBRARY_CORRELATION',max(corr) if corr and all(np.isfinite(corr)) else 'MISSING')
factor.to_pickle('scripts/miner_1_20260730_trend_consistency20_signal.pkl')
