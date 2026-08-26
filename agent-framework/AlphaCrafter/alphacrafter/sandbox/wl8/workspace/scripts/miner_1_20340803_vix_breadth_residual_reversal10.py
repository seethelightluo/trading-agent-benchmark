import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
C=pd.DataFrame({s:x.close.astype(float).replace(0,np.nan) for s,x in P.items()}).sort_index().groupby(level=0).last().loc[:'2034-08-02']
V=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(C.index).ffill()
R=np.log(C).diff(); res=R.sub(R.mean(axis=1),axis=0)
base=res.rolling(5,min_periods=4).mean().shift(1); vol=res.rolling(20,min_periods=12).std().shift(1).replace(0,np.nan)
vix=(V/V.rolling(252,min_periods=100).median()).shift(1).clip(.6,1.8)
equity=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']; breadth=R[equity].rolling(20,min_periods=15).mean().gt(0).mean(axis=1).shift(1)
gate=(vix*(.75+.75*(1-breadth))).clip(.5,2.0)
raw=(-base/vol).mul(gate,axis=0); f=raw.rank(axis=1,pct=True); f=f.sub(f.mean(axis=1),axis=0)
def q(h): return np.log(C.shift(-h)/C)
def calc(x):
 a=[];n=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],x.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z));ds.append(d)
 return pd.Series(a,index=ds),pd.Series(n,index=ds)
i,nn=calc(q(10));print('end',C.index.max().date(),'dates',len(i),'avgN',nn.mean(),'coverage',nn.mean()/15);print('IC',i.mean(),'ICIR',i.mean()/i.std(ddof=1),'hit',(i>0).mean(),'turnover',f.diff().abs().mean(axis=1).mean())
for w in [365,750,1260]:x=i.tail(w);print('recent',w,x.mean(),x.mean()/x.std(ddof=1))
for h in [1,5,20]:print('decay',h,calc(q(h))[0].mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340803_vix_breadth_residual_reversal10_signal.csv',index=False);i.rename('ic').to_csv('scripts/miner_1_20340803_vix_breadth_residual_reversal10_ic.csv')
