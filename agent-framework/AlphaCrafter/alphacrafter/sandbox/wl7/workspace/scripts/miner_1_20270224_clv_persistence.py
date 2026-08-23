import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; series={}
for a in assets:
 f=('../persistent/stock_data/' if glob.glob('../persistent/stock_data/'+a+'.csv') else '../persistent/index_data/')+a+'.csv'; d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); clv=(2*d.close-d.high-d.low)/(d.high-d.low).replace(0,np.nan); r=np.log(d.close).diff(); sig=r.rolling(20,min_periods=15).std(); s=(clv.rolling(5,min_periods=5).mean()/sig).shift(1); series[a]=pd.DataFrame({'s':s,'f1':d.close.pct_change().shift(-1),'f5':d.close.pct_change(5).shift(-5),'f10':d.close.pct_change(10).shift(-10)})
rows=[]
for dt in sorted(set().union(*[x.index for x in series.values()])):
 q=[x.loc[dt].values for x in series.values() if dt in x.index and np.isfinite(x.loc[dt]).all()]
 if len(q)>=8: rows.append([dt,len(q)]+[spearmanr(np.array(q)[:,0],np.array(q)[:,j]).statistic for j in (1,2,3)])
r=pd.DataFrame(rows,columns=['date','n','ic1','ic5','ic10']).set_index('date'); print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
for h in ['ic1','ic5','ic10']:
 x=r[h]; print(h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4)); print([(p,round((z:=r.loc[(r.index.year>=a)&(r.index.year<=b),h]).mean(),6),round(z.mean()/z.std(),4)) for p,a,b in [('20-22',2020,2022),('23-24',2023,2024),('25-27',2025,2027)]])
print('turnover', 'not computed'); r.to_csv('scripts/miner_1_20270224_clv_persistence_signal.csv')
