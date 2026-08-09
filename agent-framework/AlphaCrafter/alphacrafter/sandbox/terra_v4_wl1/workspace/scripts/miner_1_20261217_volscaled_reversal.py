import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
p=pd.concat({s:L(s) for s in U},axis=1).sort_index(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); f=-r.rolling(5,min_periods=5).sum()/vol
fw=p.pct_change().shift(-1)
def run(h):
 z=[]
 for d in f.index:
  a=f.loc[d];b=p.pct_change(h).shift(-h).loc[d]; q=pd.concat([a,b],axis=1).dropna()
  if len(q)>=8:z.append((d,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
 x=pd.DataFrame(z,columns=['date','ic','n']).set_index('date'); ic=x.ic
 print(h,'dates',len(x),'n',x.n.mean(),'cov',x.n.mean()/15,'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
 print(x.assign(y=x.index.year).groupby('y').ic.mean().round(4).to_dict())
run(1);run(5);run(10)
# turnover rank changes
print('turnover unavailable')
