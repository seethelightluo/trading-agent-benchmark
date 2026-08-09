"""miner_2: 20-observation downside energy share, tested through 2026-09-23."""
import json, glob
import numpy as np
import pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2026-09-23'; H=(1,5,10,20)
F={}; FW={}; L={k:{} for k in ['miner_3_risk_adjusted_trend_20d','miner_3_relative_volume_participation_20d','miner_1_ravmom_20obs','miner_1_volnorm_reversal_5obs','miner_1_vol_of_vol_cv20']}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 p=d.close.astype(float); r=p.pct_change(); v=d.volume.astype(float)
 # Fraction of recent return energy contributed by losses. High = losses dominate realized risk.
 F[a]=np.minimum(r,0).pow(2).rolling(20,min_periods=15).sum()/r.pow(2).rolling(20,min_periods=15).sum()
 for h in H: FW[a,h]=p.shift(-h)/p-1
 trend=(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std()
 L['miner_3_risk_adjusted_trend_20d'][a]=trend; L['miner_1_ravmom_20obs'][a]=trend
 L['miner_3_relative_volume_participation_20d'][a]=np.log(v/v.rolling(20,min_periods=15).mean())
 L['miner_1_volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
 rv=r.rolling(5,min_periods=4).std(); L['miner_1_vol_of_vol_cv20'][a]=rv.rolling(20,min_periods=15).std()/rv.rolling(20,min_periods=15).mean()
f=pd.DataFrame(F).sort_index()
def evaluate(h):
 y=pd.DataFrame({a:FW[a,h] for a in A}).reindex(f.index); rows=[]; cov=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('factor'),y.loc[dt].rename('forward')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.factor.corr(z.forward,method='spearman')));cov.append(len(z)/15)
 ic=pd.Series(dict(rows)); sd=ic.std(ddof=1)
 ranks=f.rank(axis=1,pct=True); turns=[]
 for i in range(1,len(ranks)):
  z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_std':float(sd),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_hit_ratio':float((ic>0).mean()),'ic_dates':len(ic),'mean_valid_instruments_per_ic_date':float(np.mean(cov)*15),'mean_cross_sectional_coverage':float(np.mean(cov)),'panel_cell_coverage':float(f.notna().mean().mean()),'mean_rank_turnover':float(np.mean(turns))}
print('FACTOR downside_energy_share_20obs: sum(min(return,0)^2,20)/sum(return^2,20)')
print('period',f.index.min().date(),f.index.max().date(),'universe',len(A),'signal_cells',int(f.notna().sum().sum()),'of',f.size)
M={}
for h in H:
 ic,m=evaluate(h); M[h]=m; print('HORIZON',h,json.dumps(m))
 for lab,mask in [('2020',ic.index<'2021-01-01'),('2021_2022',(ic.index>='2021-01-01')&(ic.index<'2023-01-01')),('2023_2024',(ic.index>='2023-01-01')&(ic.index<'2025-01-01')),('2025_2026',ic.index>='2025-01-01')]:
  x=ic[mask]; print('REGIME',h,lab,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
mx=0
for n,x in L.items():
 z=pd.concat([f.stack().rename('new'),pd.DataFrame(x).stack().rename('old')],axis=1).dropna(); rho=z.new.corr(z.old,method='spearman');mx=max(mx,abs(rho));print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'library_json_count',len(glob.glob('factors/*.json')))
print('DECAY',json.dumps({str(h):{'ic':M[h]['daily_paper_ic'],'icir':M[h]['daily_paper_icir'],'dates':M[h]['ic_dates']} for h in H}))
