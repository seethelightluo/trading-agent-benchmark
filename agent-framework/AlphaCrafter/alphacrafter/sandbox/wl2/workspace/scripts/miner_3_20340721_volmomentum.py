import pandas as pd,numpy as np,glob
assets=sorted([x.rsplit('/',1)[-1][:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
w=pd.concat({a:pd.to_numeric(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'],errors='coerce') for a in assets},axis=1).sort_index()
r=w.pct_change(); vol=r.rolling(20,min_periods=15).std(); factor=(w.pct_change(20)/(vol+1e-12)).shift(1); fr=w.pct_change(10).shift(-10)
rows=[]
for d in factor.index:
 ok=factor.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,factor.loc[d,ok].corr(fr.loc[d,ok]),int(ok.sum())))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); m=ic.ic.mean(); ir=m/ic.ic.std(ddof=1)*np.sqrt(252/10)
print('assets',len(assets),'dates',len(w),'valid_dates',len(ic),'mean_n',ic.n.mean(),'IC',m,'ICIR',ir,'hit',(ic.ic>0).mean(),'coverage',factor.notna().mean().mean(),'turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lab,lo,hi in [('2020-2025','2020','2025-12-31'),('2026-2029','2026','2029-12-31'),('2030-2032','2030','2032-12-31'),('2033-2034','2033','2034-12-31')]:
 a=ic.loc[(ic.index>=lo)&(ic.index<=hi),'ic'];print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252/10) if len(a)>1 else np.nan)
factor.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/miner_3_20340721_volmomentum_signal.csv',index=False)
ic.to_csv('../persistent/miner_3_20340721_volmomentum_ic.csv')
