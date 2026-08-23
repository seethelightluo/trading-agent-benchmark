import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']);D[s]=x[['date','close']].drop_duplicates('date').set_index('date').sort_index()
P=pd.DataFrame({s:D[s].close.astype(float) for s in D}).sort_index().loc[:'2034-09-27'].ffill();r=P.pct_change();v20=r.rolling(20,min_periods=15).std();v5=r.rolling(5,min_periods=5).std(); tr=(v5.mean(1)/v20.mean(1)).clip(.5,2.5)
# Gate the 5-day contrarian shock to elevated short/medium volatility transition.
F=(-(P/P.shift(5)-1)/(v20*np.sqrt(5)).replace(0,np.nan)).mul((tr>1.05).astype(float),axis=0).shift(1);F.to_csv('scripts/miner_2_20340928_transition_gated_reversal_10d_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fw=P.shift(-h)/P-1;a=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].rank().corr(z.iloc[:,1].rank());
   if pd.notna(q):a.append(q);ns.append(len(z))
 a=np.array(a);print(f'{h}D dates={len(a)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={np.mean(a):.8f} ICIR={np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)):.8f} hit={np.mean(a>0):.4f}')
print('gate_fraction',float((tr>1.05).mean()))
