import pandas as pd, numpy as np, glob
assets=sorted([x.rsplit('/',1)[-1][:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
wide=pd.concat({a:pd.to_numeric(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'],errors='coerce') for a in assets},axis=1).sort_index()
vix=pd.to_numeric(pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'],errors='coerce').reindex(wide.index).ffill()
r5=wide.pct_change(5); rv20=wide.pct_change().rolling(20,min_periods=15).std()
med=vix.rolling(60,min_periods=40).median(); sd=vix.rolling(60,min_periods=40).std()
stress=((vix-med)/(sd+1e-12)).clip(-1,2)
# axis=0 is essential: macro series scales rows, not columns.
mult=(1+0.75*stress)*(vix>med)
factor=(-(r5/(rv20+1e-12))).mul(mult,axis=0).shift(1)
fr=wide.pct_change(10).shift(-10)
rows=[]
for d in factor.index:
 x=factor.loc[d]; y=fr.loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((d,x[ok].corr(y[ok]),int(ok.sum())))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=ic.ic.mean(); ir=mean/ic.ic.std(ddof=1)*np.sqrt(252/10)
turn=factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
print('assets',len(assets),'dates',len(wide),'valid_dates',len(ic),'mean_n',ic.n.mean(),'IC',mean,'ICIR',ir,'hit',(ic.ic>0).mean(),'turnover',turn,'coverage',factor.notna().mean().mean())
for lab,lo,hi in [('2020-2025','2020','2025-12-31'),('2026-2029','2026','2029-12-31'),('2030-2032','2030','2032-12-31'),('2033-2034','2033','2034-12-31')]:
 a=ic.loc[(ic.index>=lo)&(ic.index<=hi),'ic']; print(lab,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252/10) if len(a)>1 else np.nan)
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/miner_3_20340721_stress_reversal_signal.csv',index=False)
ic.to_csv('../persistent/miner_3_20340721_stress_reversal_ic.csv')
