"""Validate one factor: 20-observation directional persistence.
High values mean a larger fraction of recent daily moves were positive, independent
of magnitude. This distinguishes persistent trends from one-off jumps.
All input is truncated at 2026-07-29 (the completed day before current date).
"""
import pandas as pd, numpy as np
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-07-29')
p={}; v={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index()
 p[a]=pd.to_numeric(d.close,errors='coerce'); v[a]=pd.to_numeric(d.get('volume'),errors='coerce')
px=pd.DataFrame(p); r=px.pct_change(fill_method=None)
# Candidate: mean of daily signs, requires 15 of 20 native observations.
sig=np.sign(r).rolling(20,min_periods=15).mean()
vol20=r.rolling(20,min_periods=15).std(); vol5=r.rolling(5,min_periods=4).std()
vv=pd.DataFrame(v)
lib={
 'miner_3_risk_adjusted_trend_20d':(px/px.shift(20)-1)/vol20,
 'miner_1_ravmom_20obs':(px/px.shift(20)-1)/vol20,
 'miner_1_volnorm_reversal_5obs':-(px/px.shift(5)-1)/vol5,
 'miner_2_realized_volatility_20obs':vol20,
 'miner_3_relative_volume_participation_20d':np.log(vv/vv.rolling(20,min_periods=15).mean())}
def ics(h):
 fw=px.shift(-h)/px-1; out=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 return pd.Series(out),ns
for h in (1,5,10,20):
 s,n=ics(h); print('H',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'dates',len(s),'mean_names',round(np.mean(n),2))
print('coverage',round(sig.notna().mean().mean(),4),'rank_turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
# Regime results for the selected 10d decision-compatible horizon.
fw=px.shift(-10)/px-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
ic=pd.DataFrame(rows,columns=['date','ic']).set_index('date')
for name,a,b in [('2020','2020-01-01','2020-12-31'),('2021_22','2021-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31')]:
 x=ic.loc[a:b,'ic']; print('REGIME',name,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'dates',len(x))
for name,x in lib.items():
 z=pd.concat([sig.stack().rename('signal'),x.stack().rename('library')],axis=1).dropna()
 print('LIBCORR',name,'rho',round(z.signal.corr(z.library,method='spearman'),6),'cells',len(z))
print('LATEST',px.index.max().date(),'assets',len(ASSETS))
