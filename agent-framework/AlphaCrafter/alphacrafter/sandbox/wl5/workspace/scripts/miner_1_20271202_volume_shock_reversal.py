import pandas as pd,numpy as np,warnings
warnings.filterwarnings('ignore')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];F=[];Y=[]
for s in syms:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',usecols=['date','close','volume']);x.date=pd.to_datetime(x.date);x=x.set_index('date').sort_index();r=x.close.pct_change(5);vs=x.volume/x.volume.rolling(20,min_periods=10).median();v=x.close.pct_change().rolling(20,min_periods=15).std();F.append((-r*np.log(vs.clip(lower=.25))/v.clip(lower=.002)).rename(s));Y.append((x.close.shift(-10)/x.close-1).rename(s))
fp=pd.concat(F,axis=1);yp=pd.concat(Y,axis=1);rows=[]
for dt in fp.index:
 z=pd.concat([fp.loc[dt],yp.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
a=pd.Series(dict(rows));print('dates',len(a),'avg_names',len(fp.columns),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4));print('regimes',[(y,round(a[a.index.year==y].mean(),4),int((a.index.year==y).sum())) for y in sorted(a.index.year.unique())]);fp.stack().rename('signal').reset_index().rename(columns={'level_1':'asset'}).to_csv('scripts/miner_1_20271202_volume_shock_reversal_signal.csv',index=False)
