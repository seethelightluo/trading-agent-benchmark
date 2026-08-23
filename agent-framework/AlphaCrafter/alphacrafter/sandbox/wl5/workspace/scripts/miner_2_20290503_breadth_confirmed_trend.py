import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
px=pd.DataFrame(D).sort_index().ffill().loc[:'2029-05-03']; r=px.pct_change()
trend=px.pct_change(60); vol=r.rolling(20).std()*np.sqrt(252)
breadth=(r.rolling(20).sum()>0).mean(axis=1); gate=(breadth-0.5)*2
sig=(trend/vol).mul(gate,axis=0).replace([np.inf,-np.inf],np.nan).rolling(3,min_periods=1).mean()
for h in [1,5,10,15]:
 fwd=px.shift(-h)/px-1; vals=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); ic=z.ic.mean(); sd=z.ic.std(ddof=1)
 print('H',h,'IC',round(ic,6),'ICIR',round(ic/sd*np.sqrt(252),6),'hit',round((z.ic>0).mean(),4),'obs',len(z),'nmean',round(z.n.mean(),2))
 if h==5: sig.to_csv('scripts/miner_2_20290503_breadth_confirmed_trend_signal.csv',index_label='date'); print('period',z.index.min(),z.index.max())
rank=sig.rank(axis=1,pct=True)
print('coverage',round(sig.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'dates',len(sig),'assets',len(sig.columns))
