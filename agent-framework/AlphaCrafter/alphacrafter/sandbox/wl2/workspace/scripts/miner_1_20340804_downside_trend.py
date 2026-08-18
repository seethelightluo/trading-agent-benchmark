import pandas as pd,numpy as np,glob
files=glob.glob('../persistent/stock_data/*.csv')
assets=sorted([x.rsplit('/',1)[-1][:-4] for x in files])
w=pd.concat({a:pd.to_numeric(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'],errors='coerce') for a in assets},axis=1).sort_index()
r=w.pct_change(); down=r.where(r<0,0.0); ds=down.rolling(30,min_periods=20).std(); f=(w.pct_change(30)/(ds+1e-12)).shift(1); fr=w.pct_change(10).shift(-10)
rows=[]
for d in f.index:
 ok=f.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,f.loc[d,ok].corr(fr.loc[d,ok]),int(ok.sum())))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');
print('assets',len(assets),'dates',len(w),'valid_dates',len(ic),'mean_n',round(ic.n.mean(),2),'IC',ic.ic.mean(),'ICIR',ic.ic.mean()/ic.ic.std(ddof=1)*np.sqrt(252/10),'hit',(ic.ic>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lab,lo,hi in [('2020-2024','2020','2024-12-31'),('2025-2029','2025','2029-12-31'),('2030-2032','2030','2032-12-31'),('2033-2034','2033','2034-12-31')]:
 a=ic.loc[(ic.index>=lo)&(ic.index<=hi),'ic'];print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252/10) if len(a)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/miner_1_20340804_downside_trend_signal.csv',index=False)
ic.to_csv('../persistent/miner_1_20340804_downside_trend_ic.csv')
