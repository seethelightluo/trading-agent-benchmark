import pandas as pd, numpy as np, glob, json, os
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-01-27')
def load(s, root='../persistent/stock_data'):
 p=f'{root}/{s}.csv'; x=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float); return x[x.index<=END]
P=pd.concat({s:load(s) for s in AS},axis=1).sort_index(); V=load('VIX','../persistent/index_data')
# Full daily panel prevents forward contamination and defines usable cells
R=P.pct_change(); vr=V.pct_change().reindex(P.index)
def csres(y, xs):
 out=pd.Series(np.nan,index=y.index); ok=y.notna()
 for x in xs: ok &= x.notna()
 if ok.sum()>=8:
  X=np.column_stack([np.ones(ok.sum())]+[x[ok].values for x in xs]); out.loc[ok]=y[ok]-X@np.linalg.lstsq(X,y[ok],rcond=None)[0]
 return out
def each(calc): return pd.DataFrame({s:calc(s) for s in AS})
vol=R.rolling(20,min_periods=15).std(); trend=P/P.shift(20)-1; rav=trend/vol
# candidate: VIX-up conditional negative beta, residualized against ordinary VIX beta, vol, trend
# high VIX days only, need 8 within 20
cond=pd.DataFrame(index=P.index,columns=AS,dtype=float); uncond=cond.copy()
for s in AS:
 for t in range(20,len(P)):
  ix=P.index[t-20:t]; a=R.loc[ix,s]; b=vr.loc[ix]; ok=a.notna()&b.notna(); up=ok&(b>0)
  if up.sum()>=8 and b[up].var()>0: cond.loc[P.index[t],s]=-np.cov(a[up],b[up],ddof=1)[0,1]/b[up].var()
  if ok.sum()>=15 and b[ok].var()>0: uncond.loc[P.index[t],s]=-np.cov(a[ok],b[ok],ddof=1)[0,1]/b[ok].var()
C=pd.DataFrame({d:csres(cond.loc[d], [uncond.loc[d],vol.loc[d],rav.loc[d]]) for d in P.index}).T
# reconstructed library signals
L={}
L['miner_3_risk_adjusted_trend_20d']=rav
L['miner_1_ravmom_20obs']=rav
L['miner_1_volnorm_reversal_5obs']=-(P/P.shift(5)-1)/vol
L['miner_3_relative_volume_participation_20d']=pd.concat({s:(pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date')['volume'].astype(float).reindex(P.index)/pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).set_index('date')['volume'].astype(float).reindex(P.index).rolling(20,min_periods=15).mean()) for s in AS},axis=1)
L['miner_1_vol_of_vol_cv20']= -(vol.rolling(20,min_periods=15).std()/vol.rolling(20,min_periods=15).mean())
# VIX residual
vb=uncond
L['miner_1_residualized_vix_stress_resilience_beta20']=pd.DataFrame({d:csres(vb.loc[d],[vol.loc[d]]) for d in P.index}).T
# beta improvement: - current downside VIX beta relative longer beta approximate
# construct down beta asset vs equally weighted market
m=R.mean(axis=1)
def beta_window(w,down=False):
 z=pd.DataFrame(index=P.index,columns=AS,dtype=float)
 for s in AS:
  for t in range(w,len(P)):
   a=R.iloc[t-w:t][s];b=m.iloc[t-w:t];ok=a.notna()&b.notna() & ((b<0) if down else True)
   if ok.sum()>=max(8,int(.7*w)) and b[ok].var()>0:z.iloc[t,z.columns.get_loc(s)]=np.cov(a[ok],b[ok],ddof=1)[0,1]/b[ok].var()
 return z
L['miner_2_downside_beta_improvement_120_20']=-(beta_window(20,True)-beta_window(120,True))
# drawdown sync / market sync approximate
DD=P/P.rolling(60,min_periods=45).max()-1
L['miner_2_drawdown_synchronization_improvement_60_20']=-(DD.rolling(20,min_periods=15).corr(DD.mean(axis=1)))
L['miner_2_market_synchronization_increase_60_20']=R.rolling(20,min_periods=15).corr(m) - R.rolling(60,min_periods=45).corr(m)
# recovery residual
raw=(P/P.shift(10)-1)*(-np.minimum(DD,0)); L['miner_1_residualized_drawdown_recovery_60_10']=pd.DataFrame({d:csres(raw.loc[d],[rav.loc[d],vol.loc[d]]) for d in P.index}).T
# tail containment residual
loss=R.where(R<0).abs().rolling(20,min_periods=8).mean(); raw=-loss/vol; L['miner_1_residualized_downside_tail_containment_20']=pd.DataFrame({d:csres(raw.loc[d],[rav.loc[d],vol.loc[d]]) for d in P.index}).T
# IC metrics
print('candidate VIX-up conditional resilience residual; visible through',END.date())
for h in [1,5,10,20]:
  fwd=P.shift(-h)/P-1; vals=[]; ns=[]
  for d in P.index:
   x=C.loc[d];y=fwd.loc[d];ok=x.notna()&y.notna()
   if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum())
  a=np.array(vals); print(h,'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan,'n',len(a),'hit',(a>0).mean() if len(a) else np.nan,'meanN',np.mean(ns) if ns else np.nan)
# coverage and turnover
print('coverage',C.notna().mean().mean(),'turnover',np.nanmean([spearmanr(C.iloc[i-1][C.iloc[i-1].notna()&C.iloc[i].notna()],C.iloc[i][C.iloc[i-1].notna()&C.iloc[i].notna()]).statistic for i in range(1,len(C)) if (C.iloc[i-1].notna()&C.iloc[i].notna()).sum()>=8]))
# correlations pooled cells
cors={}
for n,x in L.items():
 a=C.stack();b=x.stack();z=pd.concat([a,b],axis=1).dropna(); cors[n]=(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)) if len(z)>=8 else (np.nan,len(z))
print('correlations');[print(n,round(v[0],6),v[1]) for n,v in cors.items()]; print('MAX',max(abs(v[0]) for v in cors.values() if np.isfinite(v[0])))
# regime primary 20
for yr in ['2020-2024','2025','2026','2027']:
 lo,hi=({'2020-2024':('2020-01-01','2024-12-31'),'2025':('2025-01-01','2025-12-31'),'2026':('2026-01-01','2026-12-31'),'2027':('2027-01-01','2027-12-31')}[yr]); fwd=P.shift(-20)/P-1; a=[]
 for d in P.loc[lo:hi].index:
  ok=C.loc[d].notna()&fwd.loc[d].notna()
  if ok.sum()>=8:a.append(spearmanr(C.loc[d][ok],fwd.loc[d][ok]).statistic)
 print('regime',yr,'n',len(a),'ic',np.mean(a) if a else np.nan,'icir',np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan)
