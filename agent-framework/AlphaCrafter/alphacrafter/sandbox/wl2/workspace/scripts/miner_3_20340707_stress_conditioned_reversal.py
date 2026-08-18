import pandas as pd, numpy as np, glob
assets=[x.rsplit('/',1)[-1][:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
wide=pd.concat({a:pd.to_numeric(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'],errors='coerce') for a in assets},axis=1).sort_index().astype(float)
vix=pd.to_numeric(pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'],errors='coerce').reindex(wide.index).ffill()
r=wide.pct_change(5); vol=wide.pct_change().rolling(20).std(); med=vix.rolling(60,min_periods=40).median(); vz=(vix-med)/(vix.rolling(60,min_periods=40).std()+1e-12)
factor=(-(r/(vol+1e-12))*(1+0.75*vz.clip(-1,2))*(vix>med)).shift(1).astype(float)
for h in [1,5,10,20]:
 fr=wide.pct_change(h).shift(-h); xr=factor.rank(axis=1); yr=fr.rank(axis=1); ic=xr.corrwith(yr,axis=1,method='pearson').dropna(); ic=ic[xr.notna().sum(axis=1).loc[ic.index]>=8]
 mean=ic.mean(); ir=mean/ic.std(ddof=1)*np.sqrt(252/h); turn=factor.diff().abs().div(2).mean(axis=1).dropna().mean()
 print('horizon',h,'dates',len(ic),'mean_n',xr.notna().sum(axis=1).loc[ic.index].mean(),'IC',mean,'ICIR',ir,'hit',(ic>0).mean(),'turn',turn)
 for lab,lo,hi in [('2026-2028','2026','2028-12-31'),('2029-2031','2029','2031-12-31'),('2032-now','2032','2034-12-31')]:
  a=ic[(ic.index>=lo)&(ic.index<=hi)]; print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252/h))
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/miner_3_20340707_stress_conditioned_reversal_signal.csv',index=False)
print('assets',len(assets),'dates',len(wide),'artifact_rows',len(out))
