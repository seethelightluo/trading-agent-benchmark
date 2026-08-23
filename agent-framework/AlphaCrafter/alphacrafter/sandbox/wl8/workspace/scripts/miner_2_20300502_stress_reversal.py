import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-05-01'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); D[s]=x.loc[:cut,'close']
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change(); disp=r.rolling(20).std().mean(axis=1); z=(disp-disp.rolling(252,min_periods=80).mean())/disp.rolling(252,min_periods=80).std()
# continuous stress-gated reversal: 5d reversal strengthened in high dispersion, suppressed in calm markets
stress=(1+0.8*np.tanh(z)).clip(.2,1.8); sig=(-px.pct_change(5)).mul(stress,axis=0).shift(1); rows=[]
for i in range(len(px)-10):
 if px.index[i]>cut-pd.Timedelta(days=15): break
 f=sig.iloc[i]; y=px.iloc[i+10]/px.iloc[i]-1; ok=f.notna()&y.notna()
 if ok.sum()>=8: rows.append((px.index[i],spearmanr(f[ok],y[ok]).statistic,ok.sum()))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name,q in [('full',out),('2020_23',out.loc['2020':'2023']),('2024_26',out.loc['2024':'2026-07-15']),('post',out.loc['2026-07-16':'2028-12-31']),('recent',out.loc['2029-01-01':]),('recent180',out.tail(180))]:
 a=q.ic; print(name,len(a),round(q.n.mean(),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
print('coverage',round(sig.notna().mean().mean(),5),'turnover',round(sig.rank(pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
sig.to_csv('scripts/miner_2_20300502_stress_reversal_signal.csv',index_label='date'); out.to_csv('scripts/miner_2_20300502_stress_reversal_ic.csv')
