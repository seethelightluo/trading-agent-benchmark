import numpy as np,pandas as pd,glob,json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2029-05-02'; ROOT='../persistent/stock_data'
D={a:pd.read_csv(f'{ROOT}/{a}.csv').set_index('date').sort_index().loc[:END] for a in A}; ix=sorted(set().union(*[set(x.index) for x in D.values()])); close=pd.DataFrame({a:D[a].reindex(ix).close for a in A}); volume=pd.DataFrame({a:D[a].reindex(ix).volume for a in A}); high=pd.DataFrame({a:D[a].reindex(ix).high for a in A}); low=pd.DataFrame({a:D[a].reindex(ix).low for a in A}); op=pd.DataFrame({a:D[a].reindex(ix).open for a in A}); r=close.pct_change(fill_method=None); med=r.median(axis=1); disp=r.std(axis=1)
def macro(s): return pd.read_csv('../persistent/index_data/'+s+'.csv').set_index('date').sort_index().reindex(ix).close.pct_change(fill_method=None)
vix=macro('VIX');dxy=macro('DXY')
def beta(x,y,w=60,minp=15):return x.rolling(w,min_periods=minp).cov(y).div(y.rolling(w,min_periods=minp).var(),axis=0)
def corr(x,y,w=60,minp=15):return x.rolling(w,min_periods=minp).corr(y)
# Candidate: tail-VIX response loading in high systemic-dispersion days minus low-dispersion days.
tail=(vix.abs()>vix.abs().rolling(60,min_periods=30).quantile(.8)); ds=disp.rolling(20,min_periods=12).mean(); state=ds>ds.rolling(60,min_periods=30).median(); shock=vix.shift(1)*tail.shift(1)
def conditional_beta(mask):
 y=shock.where(mask); return r.rolling(60,min_periods=25).cov(y).div(y.rolling(60,min_periods=25).var(),axis=0)
f=conditional_beta(state)-conditional_beta(~state)
vis=pd.Index(ix)
print('FACTOR vix_tail_dispersion_state_response_spread_60obs endpoint',vis[-1],'assets',len(A),'raw_cells',int(f.notna().sum().sum()),'of',len(vis)*15)
def stats(sub,h):
 fw=close.shift(-h).div(close)-1; vals=[];ns=[];turn=[];prev=None
 for t in sub:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  q=f.loc[t].rank(); z=pd.concat([q,prev],axis=1).dropna() if prev is not None else pd.DataFrame()
  if len(z)>=8:turn.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(vals);return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns),np.mean(turn)
for h in [1,5,10,20]:
 x=stats(vis,h);print('H',h,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4),'mean_n',round(x[4],2),'ic_date_coverage',round(x[0]/len(vis),4),'turn',round(x[5],6))
for lab,sub in [('2020_21',vis[vis<'2022-01-01']),('2022_23',vis[(vis>='2022-01-01')&(vis<'2024-01-01')]),('2024_25',vis[(vis>='2024-01-01')&(vis<'2026-01-01')]),('2026_current',vis[vis>='2026-01-01'])]:
 x=stats(sub,20);print('REGIME',lab,'dates',x[0],'IC',round(x[1],6),'ICIR',round(x[2],6),'hit',round(x[3],4))
# Reconstruct all admitted signal families for full diversification evidence.
lib={}
lib['dxy_median_trend_regime_beta_spread_60obs']=beta(r,dxy.where(med.rolling(20).median()>0))-beta(r,dxy.where(med.rolling(20).median()<=0))
neg=r<0; lv=np.log(volume.where(volume>0));
lib['inverted_downside_volume_participation_acceleration_20_60obs']=-(np.log(volume.where(neg).rolling(60,min_periods=20).mean()/volume.where(~neg).rolling(60,min_periods=20).mean())-np.log(volume.where(neg).rolling(20,min_periods=10).mean()/volume.where(~neg).rolling(20,min_periods=10).mean()))
lib['vix_tail_lagged_response_persistence_60obs']=beta(r,shock,60,25)
lib['miner_3_relative_volume_participation_20d']=np.log(volume/volume.rolling(20,min_periods=12).mean())
lib['dxy_relative_vol_regime_beta_spread_60obs']=beta(r,dxy.where(r.rolling(20).std().gt(r.rolling(20).std().rolling(60).median())))-beta(r,dxy.where(r.rolling(20).std().le(r.rolling(20).std().rolling(60).median())))
lib['miner_3_return_persistence_autocorr_20obs']=r.rolling(20,min_periods=15).corr(r.shift(1));lib['dxy_shock_lagged_response_persistence_60obs']=beta(r,dxy.shift(1).where(dxy.shift(1).abs()>dxy.abs().rolling(60).quantile(.8)),60,25);lib['miner_3_risk_adjusted_trend_20d']=(close/close.shift(20)-1)/r.rolling(20,min_periods=15).std();lib['adaptive_vix_relief_beta_change_25_60obs']=beta(r,vix.where(vix<0),25,12)-beta(r,vix.where(vix<0),60,15);lib['vix_shock_relief_beta_asymmetry_60obs']=beta(r,vix.where(vix>0))-beta(r,vix.where(vix<0));lib['vol_orthogonal_median_beta_60obs']=beta(r,med).sub(beta(r,med).median(axis=1),axis=0);lib['miner_1_correlation_asymmetry_60obs']=corr(r.where(med<0),med.where(med<0))-corr(r.where(med>=0),med.where(med>=0));lib['volatility_clustering_autocorr_20obs']=r.abs().rolling(20,min_periods=15).corr(r.abs().shift(1));lib['dispersion_sensitivity_20obs']=corr(r,disp,20);lib['miner_1_return_sign_balance_20obs']=(r>0).rolling(20,min_periods=15).mean()-.5;lib['miner_2_realized_volatility_20obs']=r.rolling(20,min_periods=15).std();lib['miner_1_volnorm_reversal_5obs']=-(close/close.shift(5)-1)/r.rolling(5,min_periods=4).std();lib['miner_3_risk_adjusted_trend_acceleration_20_60d']=(close/close.shift(20)-1)/r.rolling(20).std()-(close/close.shift(60)-1)/r.rolling(60).std();lib['miner_3_return_directional_efficiency_20obs']=r.rolling(20).sum().abs()/r.abs().rolling(20).sum();lib['miner_3_relative_liquidity_stress_20_60obs']=np.log((r.abs()/volume).rolling(20).mean()/(r.abs()/volume).rolling(60).mean())
# Exact reconstruction unavailable for a few residual/range/OHLC factors, use their direct standard definitions.
lib['inverted_dispersion_regime_range_state_20_60obs']=-np.log(((high-low)/close).rolling(20).mean()/((high-low)/close).rolling(60).mean())*np.sign(np.log(r.rolling(20).std().median(axis=1)/r.rolling(20).std().median(axis=1).rolling(60).median()))
lib['overnight_daytime_reversal_concordance_20obs']=-((op/close.shift(1)-1)*((close/op)-1));lib['downside_volume_participation_asymmetry_60obs']=np.log(volume.where(neg).rolling(60).mean()/volume.where(~neg).rolling(60).mean())
print('LIBRARY_CORRELATION reconstructed',len(lib)); mx=-1;near='';pairs=0
for n,x in lib.items():
 q=[];cell=0
 for t in vis:
  z=pd.concat([f.loc[t],x.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna();cell+=len(z)
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 m=max(map(abs,q)) if q else np.nan
 print(n,'dates',len(q),'maxabs',round(m,6),'mean',round(np.mean(q),6),'paired_cells',cell)
 if np.isfinite(m) and m>mx:mx=m;near=n;pairs=cell
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'CLOSEST',near,'PAIRED_CELLS',pairs)
print('NOTE: 2 admitted residual-based signals not reconstructed; correlation evidence is incomplete and thus admission cannot be claimed unless candidate fails IC gates.')
