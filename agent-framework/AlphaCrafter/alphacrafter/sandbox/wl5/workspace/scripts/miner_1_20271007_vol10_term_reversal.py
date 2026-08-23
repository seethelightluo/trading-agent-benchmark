import os, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-10-06')
def main():
 ds={}
 for s in U:
  f='../persistent/stock_data/'+s+'.csv'
  if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
  if os.path.exists(f):
   d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].drop_duplicates('date'); ds[s]=d.set_index('date').close.astype(float)
 px=pd.DataFrame(ds).sort_index(); rets=px.pct_change(); v10=rets.rolling(10,min_periods=10).std(); v30=rets.rolling(30,min_periods=30).std(); fac=-(v10/v30)
 rows=[]
 for dt in fac.index:
  f=fac.loc[dt]; fw=px.shift(-5).loc[dt]/px.loc[dt]-1; z=pd.concat([f,fw],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 x=pd.DataFrame(rows,columns=['date','ic','n']).dropna(); ic=x.ic; ir=ic.mean()/ic.std(ddof=1)
 print('range',px.index.min(),px.index.max(),'dates',len(x),'avg_n',x.n.mean(),'assets',len(px.columns)); print('IC',ic.mean(),'ICIR',ir,'hit',(ic>0).mean(),'absIC',abs(ic.mean()),'absICIR',abs(ir)); print('year',x.assign(year=x.date.dt.year).groupby('year').ic.agg(['mean','count']).to_string()); fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20271007_vol10_term_reversal_signal.csv',index=False)
if __name__=='__main__': main()
