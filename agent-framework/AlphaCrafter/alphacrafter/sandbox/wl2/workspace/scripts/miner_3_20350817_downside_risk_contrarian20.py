import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-08-17')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change()
# Downside-risk-adjusted contrarian medium-term return: negative 20d return scaled by recent downside deviation.
down=r.where(r<0).rolling(20,min_periods=10).std(); f=(-px.pct_change(20)/down).shift(1)
def calc(h):
 fr=px.shift(-h)/px-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=d.ic
 print('h',h,'N',len(a),'avgN',round(d.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for label,x in [('2020_2025',a.loc['2020':'2025']),('2026_2028',a.loc['2026':'2028']),('2029_2032',a.loc['2029':'2032']),('2033_2035',a.loc['2033':])]:
  if len(x): print(label,'N',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 return d
D=calc(10)
print('assets',len(U),'dates',len(D),'coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4))
f.to_csv('../persistent/miner_3_20350817_downside_risk_contrarian20_signal.csv'); D.to_csv('../persistent/miner_3_20350817_downside_risk_contrarian20_ic.csv')
