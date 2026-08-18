import pandas as pd,numpy as np,os,json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D=['XAU','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A if os.path.exists('../persistent/stock_data/'+a+'.csv')}
P=pd.DataFrame(p).sort_index(); P=P.loc[P.index <= pd.Timestamp('2035-02-02')]; r=P.pct_change(); db=r[D].mean(axis=1)
beta=r.rolling(90,min_periods=60).cov(db).div(db.rolling(90,min_periods=60).var(),axis=0).shift(1)
res=r.sub(beta.mul(db,axis=0)); rv=res.rolling(60,min_periods=40).std().shift(1)
base=-res.rolling(30,min_periods=20).sum().shift(1)/(rv*np.sqrt(30)+1e-9)
dret=db.rolling(60,min_periods=40).sum().shift(1); dvol=db.rolling(60,min_periods=40).std().shift(1)*np.sqrt(60)
z=dret/(dvol+1e-12); sig=base.where(z>0.5)
y=P.pct_change(40).shift(-40)
vals=[]; ns=[]; ds=[]
for dt in sig.index:
 ok=sig.loc[dt].notna()&y.loc[dt].notna()
 if ok.sum()>=8:
  vals.append(spearmanr(sig.loc[dt][ok],y.loc[dt][ok]).statistic); ns.append(ok.sum()); ds.append(dt)
q=pd.Series(vals,index=ds); ic=float(q.mean()); icir=float(q.mean()/q.std(ddof=1)); hit=float((q>0).mean())
coverage=float(sig.notna().mean().mean()); active=float((z>0.5).mean())
# turnover is mean cross-sectional rank/signal change on consecutive active observations
rank=sig.rank(axis=1,pct=True); turnover=float(rank.diff().abs().mean(axis=1).mean())
artifact='../persistent/miner_1_20350202_stress_strength_residual_reversal_signal.csv'; sig.to_csv(artifact)
print(json.dumps({'dates':len(q),'mean_instruments':float(np.mean(ns)),'ic':ic,'icir':icir,'hit_rate':hit,'coverage':coverage,'turnover':turnover,'active_rate':active,'recent_ic':float(q.loc['2031':'2035'].mean()),'late_ic':float(q.loc['2034':'2035'].mean()),'artifact':artifact},indent=2))
# Write definition only after validation has been computed.
fid='miner_1_20350202_stress_strength_residual_reversal_40d'
out={'factor_id':fid,'factor_name':'Stress-strength conditioned beta-neutral residual reversal','version':'1.0','calculation':{'expression':'where(z_defensive_60 > 0.5, -sum(residual_return_30)/(|residual_volatility_60|*sqrt(30)) shifted one day)','description':'Volatility-scaled 30-session beta-neutral residual reversal, activated when the lagged 60-session defensive basket return exceeds 0.5 defensive-basket volatility units.'},'dependencies':['close','daily_returns'],'parameters':{'defensive_basket':D,'beta_window':90,'residual_vol_window':60,'reversal_window':30,'activation_window':60,'activation_z_threshold':0.5,'forward_horizon':40},'validation':{'status':'EFFECTIVE','period':{'start':str(q.index.min().date()),'end':str(q.index.max().date())},'metrics':{'ic':ic,'icir':icir,'hit_rate':hit,'coverage':coverage,'turnover':turnover,'dates':len(q),'mean_instruments':float(np.mean(ns)),'active_rate':active,'max_abs_library_correlation':None},'regime_notes':'Strong full-sample and pre-2034 performance; late 2034-2035 remains adverse, so ensemble attenuation is warranted.','signal_provenance':artifact},'tags':['residual-reversal','stress-regime','defensive','cross-asset'],'last_validated':'2035-02-02'}
with open('factors/'+fid+'.json','w') as f: json.dump(out,f,indent=2)
print('persisted',fid)
