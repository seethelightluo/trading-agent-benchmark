import pandas as pd, numpy as np, json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cutoff=pd.Timestamp('2032-02-04')
cl={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}
C=pd.DataFrame(cl).loc[:cutoff]; r=C.pct_change(); vol=r.rolling(40,min_periods=20).std()
# Contrarian acceleration: reverse recent 20d return relative to one-third of 60d return, risk scaled.
F=-(C/C.shift(20)-1-(C/C.shift(60)-1)/3)/(vol*np.sqrt(20)+1e-12)
def calc(h):
 out=[]
 for i in range(61,len(C)-h):
  a=F.iloc[i]; b=C.iloc[i+h]/C.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   z=spearmanr(a[ok],b[ok]).statistic
   if np.isfinite(z): out.append((C.index[i],z,int(ok.sum())))
 return pd.DataFrame(out,columns=['date','ic','n'])
res={}
for h in [5,10,20]:
 x=calc(h); m=x.ic.mean(); ir=m/(x.ic.std(ddof=1)+1e-12)*np.sqrt(len(x));
 res[h]={'dates':len(x),'avg_n':float(x.n.mean()),'coverage':float(x.n.mean()/15),'IC':float(m),'ICIR':float(ir),'hit':float((x.ic>0).mean())}
 print(f'h={h} dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={m:.7f} ICIR={ir:.4f} hit={(x.ic>0).mean():.4f}')
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
x=calc(10); print(x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string())
print('data',len(C),C.index.min().date(),C.index.max().date())
if abs(res[10]['IC'])>=.007 and abs(res[10]['ICIR'])>=.084:
 fid='miner_2_20320205_risk_scaled_acceleration_reversal_10d'
 obj={'factor_id':fid,'factor_name':'Risk-Scaled Acceleration Reversal','version':'1.0','calculation':{'expression':'-((close/close.shift(20)-1)-(close/close.shift(60)-1)/3)/(rolling_std(pct_change(close),40)*sqrt(20))','description':'Contrarian cross-asset acceleration signal: reverses recent 20-day performance relative to a one-third 60-day baseline and scales by 40-day realized volatility.'},'dependencies':['close'],'parameters':{'short_window':20,'long_window':60,'vol_window':40,'baseline_divisor':3,'primary_horizon_days':10,'min_cross_section':8},'validation':{'status':'EFFECTIVE','period':{'start':str(C.index.min().date()),'end':str(C.index.max().date())},'metrics':{'IC':res[10]['IC'],'ICIR':res[10]['ICIR'],'coverage':res[10]['coverage'],'turnover':float(F.rank(axis=1,pct=True).diff().abs().mean().mean()),'valid_dates':res[10]['dates'],'avg_instruments':res[10]['avg_n'],'hit_ratio':res[10]['hit'],'decay':res,'max_abs_library_correlation':None},'regime_notes':'Strong positive reversal association in 2023-2031, with earlier 2020-2022 momentum-like regime; interpret conservatively and audit correlation with existing reversal factors.','admission_thresholds':{'absolute_daily_paper_IC':0.007,'absolute_daily_paper_ICIR':0.084},'provenance':{'script':'scripts/miner_2_20320205_risk_scaled_acceleration_reversal.py','signal_definition':'F artifact reconstructable from OHLCV close only'}},'tags':['reversal','acceleration','risk_scaled','cross_asset'],'last_validated':'2032-02-05T00:00:00'}
 path='factors/'+fid+'.json'; open(path,'w').write(json.dumps(obj,indent=2)); chk=json.load(open(path)); print('PERSISTED',path,chk['factor_id'],chk['validation']['status'],chk['validation']['metrics']['IC'],chk['validation']['metrics']['ICIR'])
