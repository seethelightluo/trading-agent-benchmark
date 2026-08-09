import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2029-05-16';ROOT='../persistent/stock_data'
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index().loc[:END] for a in A}; ix=sorted(set().union(*[set(x.index) for x in D.values()])); C=pd.DataFrame({a:D[a].reindex(ix).close for a in A});V=pd.DataFrame({a:D[a].reindex(ix).volume for a in A});H=pd.DataFrame({a:D[a].reindex(ix).high for a in A});L=pd.DataFrame({a:D[a].reindex(ix).low for a in A});O=pd.DataFrame({a:D[a].reindex(ix).open for a in A});r=C.pct_change(fill_method=None);m=r.median(axis=1);disp=r.std(axis=1)
def mac(n):return pd.read_csv('../persistent/index_data/'+n+'.csv').set_index('date').sort_index().reindex(ix).close.pct_change(fill_method=None)
vix=mac('VIX');dxy=mac('DXY')
def beta(x,y,w=60,minp=25): return x.rolling(w,min_periods=minp).cov(y).div(y.rolling(w,min_periods=minp).var(),axis=0)
def co(x,y,w=60,minp=15):return x.rolling(w,min_periods=minp).corr(y)
# Single candidate: tail-VIX response, continuously scaled by prior systemic dispersion surprise.
tail=vix.shift(1).where(vix.shift(1).abs()>vix.abs().rolling(60,min_periods=30).quantile(.8),0)
dz=(disp.rolling(20,min_periods=15).mean()/disp.rolling(60,min_periods=40).mean()-1).clip(-3,3).shift(1)
f=beta(r,tail*dz,60,25)
print('FACTOR vix_tail_dispersion_scaled_response_60obs visible_through',ix[-1],'assets',len(A),'raw_cells',int(f.notna().sum().sum()),'of',len(ix)*15)
def st(sub,h):
 fw=C.shift(-h).div(C)-1; vals=[];ns=[];turn=[];prev=None
 for t in sub:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  if prev is not None:
   z=pd.concat([f.loc[t],prev],axis=1).dropna()
   if len(z)>=8:turn.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  prev=f.loc[t]
 x=np.array(vals); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns),np.mean(turn)
for h in [1,5,10,20]:
 z=st(ix,h);print('H',h,'dates',z[0],'IC',round(z[1],6),'ICIR',round(z[2],6),'hit',round(z[3],4),'mean_n',round(z[4],2),'turn',round(z[5],6))
for lab,sub in [('2020_21',[t for t in ix if t<'2022-01-01']),('2022_23',[t for t in ix if '2022-01-01'<=t<'2024-01-01']),('2024_25',[t for t in ix if '2024-01-01'<=t<'2026-01-01']),('2026_current',[t for t in ix if t>='2026-01-01'])]:
 z=st(sub,20);print('REGIME',lab,'dates',z[0],'IC',round(z[1],6),'ICIR',round(z[2],6),'hit',round(z[3],4))
# Exact admitted-library signal reconstruction.
lib={}; neg=r<0; sig20=r.rolling(20,min_periods=15).std(); bm=beta(r,m); ca=co(r.where(m<0),m.where(m<0))-co(r.where(m>=0),m.where(m>=0)); res=bm.sub(bm.median(axis=1),axis=0)
lib['dxy_median_trend_regime_beta_spread_60obs']=beta(r,dxy.where(m.rolling(20,min_periods=15).median()>0))-beta(r,dxy.where(m.rolling(20,min_periods=15).median()<=0))
lib['inverted_downside_volume_participation_acceleration_20_60obs']=-(np.log(V.where(neg).rolling(60,min_periods=20).mean()/V.where(~neg).rolling(60,min_periods=20).mean())-np.log(V.where(neg).rolling(20,min_periods=10).mean()/V.where(~neg).rolling(20,min_periods=10).mean()))
lib['vix_tail_lagged_response_persistence_60obs']=beta(r,tail,60,25);lib['miner_3_relative_volume_participation_20d']=np.log(V/V.rolling(20,min_periods=12).mean())
lib['dxy_relative_vol_regime_beta_spread_60obs']=beta(r,dxy.where(sig20.gt(sig20.rolling(60,min_periods=45).median())))-beta(r,dxy.where(sig20.le(sig20.rolling(60,min_periods=45).median())))
lib['miner_3_return_persistence_autocorr_20obs']=r.rolling(20,min_periods=15).corr(r.shift(1)); ds=dxy.shift(1).where(dxy.shift(1).abs()>dxy.abs().rolling(60,min_periods=30).quantile(.8));lib['dxy_shock_lagged_response_persistence_60obs']=beta(r,ds,60,25);lib['miner_3_risk_adjusted_trend_20d']=(C/C.shift(20)-1)/sig20
lib['adaptive_vix_relief_beta_change_25_60obs']=beta(r,vix.where(vix<0),25,12)-beta(r,vix.where(vix<0),60,15); lib['excess_downside_beta_ca_orthogonal_60obs']=(beta(r,m.where(m<0))-bm).sub((beta(r,m.where(m<0))-bm).cov(ca).div(ca.var())*ca,axis=0)
lib['downside_volume_participation_asymmetry_60obs']=np.log(V.where(neg).rolling(60,min_periods=20).mean()/V.where(~neg).rolling(60,min_periods=20).mean());lib['miner_3_relative_liquidity_stress_20_60obs']=np.log((r.abs()/V).rolling(20,min_periods=15).mean()/(r.abs()/V).rolling(60,min_periods=45).mean());lib['vix_shock_relief_beta_asymmetry_60obs']=beta(r,vix.where(vix>0),60,12)-beta(r,vix.where(vix<0),60,12);lib['vol_orthogonal_median_beta_60obs']=res
# residual semivol
inter=r.rolling(60,min_periods=45).mean()-bm*m.rolling(60,min_periods=45).mean();er=r-(inter+bm*m);lib['residual_downside_semivol_share_60obs']=np.sqrt(er.clip(upper=0).pow(2).rolling(60,min_periods=45).mean())/np.sqrt(er.pow(2).rolling(60,min_periods=45).mean())
lib['miner_3_return_directional_efficiency_20obs']=r.rolling(20,min_periods=15).sum().abs()/r.abs().rolling(20,min_periods=15).sum();lib['dispersion_sensitivity_20obs']=co(r,disp,20,15);lib['volatility_clustering_autocorr_20obs']=r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1));rg=(H-L).abs()/C;lib['inverted_dispersion_regime_range_state_20_60obs']=-np.log(rg.rolling(20,min_periods=15).mean()/rg.rolling(60,min_periods=45).mean()).mul(np.sign(np.log(sig20.median(axis=1)/sig20.median(axis=1).rolling(60,min_periods=45).median())).replace(0,1),axis=0);lib['miner_1_return_sign_balance_20obs']=(r>0).rolling(20,min_periods=15).mean()-.5;lib['overnight_daytime_reversal_concordance_20obs']=-((O/C.shift(1)-1)*((C/O)-1));lib['miner_2_realized_volatility_20obs']=sig20;lib['miner_1_volnorm_reversal_5obs']=-(C/C.shift(5)-1)/r.rolling(5,min_periods=4).std();lib['miner_3_risk_adjusted_trend_acceleration_20_60d']=(C/C.shift(20)-1)/sig20-(C/C.shift(60)-1)/r.rolling(60,min_periods=45).std();lib['miner_1_correlation_asymmetry_60obs']=ca
mx=-1;who=''; evidence=True
for n,x in lib.items():
 q=[]
 for t in ix:
  z=pd.concat([f.loc[t],x.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if not q:evidence=False
 a=max(map(abs,q)) if q else np.nan;print('LIB',n,'dates',len(q),'maxabs',round(a,6))
 if a>mx:mx=a;who=n
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'CLOSEST',who,'LIBRARY_FACTORS',len(lib),'EVIDENCE_COMPLETE',evidence)
