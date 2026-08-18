import pandas as pd,numpy as np,glob
files=glob.glob('../persistent/stock_data/*.csv'); assets=sorted([x.rsplit('/',1)[-1][:-4] for x in files])
w=pd.concat({a:pd.to_numeric(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'],errors='coerce') for a in assets},axis=1).sort_index(); r=w.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(w.index).ffill(); vz=((v-v.rolling(60,min_periods=30).mean())/(v.rolling(60,min_periods=30).std()+1e-12)).shift(1); f=(-w.pct_change(5)*(1+vz.clip(-1,2)*0.5)).shift(1); fr=w.pct_change(10).shift(-10)
rows=[]
for d in f.index:
 ok=f.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,f.loc[d,ok].corr(fr.loc[d,ok]),int(ok.sum())))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('assets',len(assets),'dates',len(w),'valid_dates',len(ic),'mean_n',ic.n.mean(),'IC',ic.ic.mean(),'ICIR',ic.ic.mean()/ic.ic.std(ddof=1)*np.sqrt(252/10),'hit',(ic.ic>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lab,lo,hi in [('2020-2024','2020','2024-12-31'),('2025-2029','2025','2029-12-31'),('2030-2032','2030','2032-12-31'),('2033-2034','2033','2034-12-31')]:
 a=ic.loc[(ic.index>=lo)&(ic.index<=hi),'ic']; print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252/10) if len(a)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/miner_1_20340804_vix_reversal_signal.csv',index=False); ic.to_csv('../persistent/miner_1_20340804_vix_reversal_ic.csv')
