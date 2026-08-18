import pandas as pd,numpy as np,glob
assets=sorted([x.rsplit('/',1)[-1][:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
w=pd.concat({a:pd.to_numeric(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'],errors='coerce') for a in assets},axis=1).sort_index(); r=w.pct_change()
# Trend persistence with confirmation: average of 20d and 60d returns, lagged
factor=(0.5*(w/w.shift(20)-1)+0.5*(w/w.shift(60)-1)).shift(1); fr=w.pct_change(10).shift(-10)
rows=[]
for d in factor.index:
 ok=factor.loc[d].notna()&fr.loc[d].notna()
 if ok.sum()>=8: rows.append((d,factor.loc[d,ok].corr(fr.loc[d,ok]),int(ok.sum())))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); m=ic.ic.mean(); ir=m/ic.ic.std(ddof=1)*np.sqrt(252/10)
print('candidate confirmed_20_60_momentum'); print('assets',len(assets),'valid_dates',len(ic),'mean_n',round(ic.n.mean(),3),'IC',round(m,6),'ICIR',round(ir,6),'hit',round((ic.ic>0).mean(),4),'coverage',round(factor.notna().mean().mean(),4),'turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for lab,lo,hi in [('2026-2029','2026','2029-12-31'),('2030-2032','2030','2032-12-31'),('2033-2034','2033','2034-12-31')]:
 a=ic.loc[(ic.index>=lo)&(ic.index<=hi),'ic']; print(lab,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1)*np.sqrt(252/10),6))
factor.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('../persistent/miner_3_20340818_confirmed_mom_signal.csv',index=False); ic.to_csv('../persistent/miner_3_20340818_confirmed_mom_ic.csv')
