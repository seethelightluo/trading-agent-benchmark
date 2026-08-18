import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];p={}
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']);p[s]=d.set_index('date').close
p=pd.DataFrame(p).sort_index();r=p.pct_change(); sig=(-(p.pct_change(3)/r.rolling(20).std())).shift(1); f=p.pct_change().shift(-1)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(q),'meanN',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turn',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lab,mask in [('early',q.index<'2028-01-01'),('mid',(q.index>='2028-01-01')&(q.index<'2031-01-01')),('recent',q.index>='2031-01-01')]:
 a=q.loc[mask,'ic'];print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1))
sig.to_csv('../persistent/miner_3_20341110_volscaled_reversal3_signal.csv');q.to_csv('../persistent/miner_3_20341110_volscaled_reversal3_ic.csv')
