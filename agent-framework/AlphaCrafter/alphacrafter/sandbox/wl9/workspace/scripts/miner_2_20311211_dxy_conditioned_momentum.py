import pandas as pd, numpy as np
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for a in assets}
p=pd.DataFrame(D).sort_index().ffill(); p=p[~p.index.duplicated()]; logp=np.log(p)
res20=logp.diff(20).sub(logp.diff(20).mean(axis=1),axis=0)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].reindex(p.index).ffill()
rank=dxy.pct_change(20).rolling(252,min_periods=120).apply(lambda x: np.mean(x[-1]>x[:-1]),raw=True)
f=(res20.mul(1.5-rank,axis=0)).shift(1)
def ic(a,b):
 out=[]
 for x,y in zip(a,b):
  ok=np.isfinite(x)&np.isfinite(y)
  if ok.sum()>=8: out.append(np.corrcoef(pd.Series(x[ok]).rank(),pd.Series(y[ok]).rank())[0,1])
 return np.asarray(out)
for h in [5,10,20,40,60]:
 a=ic(f.to_numpy(),(logp.shift(-h)-logp).to_numpy()); print(h,'dates',len(a),'avgN',round(f.notna().sum(axis=1).mean(),2),'coverage',round(f.notna().sum(axis=1).mean()/15,4),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'period',p.index.min().date(),p.index.max().date())
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').dropna().to_csv('scripts/miner_2_20311211_dxy_conditioned_momentum_signal.csv',index=False)
