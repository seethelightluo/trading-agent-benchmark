import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-05-01'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); D[s]=x.loc[:cut,'close'].astype(float)
px=pd.concat(D,axis=1).sort_index(); ret=px.pct_change(); disp=ret.rolling(20).std().mean(axis=1)
z=(disp-disp.rolling(252,min_periods=80).mean())/disp.rolling(252,min_periods=80).std(); adapt=(1-.35*np.tanh(z)).clip(.5,1.5)
sig=px.pct_change(20).mul(adapt,axis=0).shift(1); rows=[]
for i in range(len(px)-10):
 d=px.index[i]
 if d>cut-pd.Timedelta(days=15): break
 f=sig.iloc[i]; y=px.iloc[i+10]/px.iloc[i]-1; ok=f.notna()&y.notna()
 if ok.sum()>=8: rows.append((d,spearmanr(f[ok],y[ok]).statistic,ok.sum()))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
periods=[('full',r),('2020_23',r.loc['2020':'2023']),('2024_26',r.loc['2024':'2026-07-15']),('post',r.loc['2026-07-16':'2028-12-31']),('recent',r.loc['2029-01-01':]),('recent180',r.tail(180))]
for name,zr in periods:
 ic=zr.ic.dropna(); print(name,len(ic),round(zr.n.mean(),2),round(ic.mean(),6),round(ic.mean()/ic.std(ddof=1),6),round((ic>0).mean(),4))
print('coverage',round(sig.notna().mean().mean(),5),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'start',r.index.min(),'end',r.index.max())
sig.to_csv('scripts/miner_2_20300502_dispersion_adaptive_trend_signal.csv',index_label='date'); r.to_csv('scripts/miner_2_20300502_dispersion_adaptive_trend_ic.csv')
