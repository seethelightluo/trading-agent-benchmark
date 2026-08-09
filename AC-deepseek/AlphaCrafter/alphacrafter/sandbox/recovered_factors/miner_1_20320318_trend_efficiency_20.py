import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
# Signed 20-session trend efficiency: directional return divided by path length; rewards persistent trends while scaling noisy moves.
f=r.rolling(20,min_periods=15).sum()/r.abs().rolling(20,min_periods=15).sum()
def calc(h,idx=None):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in (p.index if idx is None else idx):
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); return len(a),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),np.mean(ns)
for h in [1,5,10,20]: print('H%d dates=%d IC=%.6f ICIR=%.6f meanN=%.2f'%(h,*calc(h)))
rank=f.rank(axis=1,pct=True); turns=[]
for i in range(10,len(rank),10):
 z=rank.iloc[i-10].dropna().index.intersection(rank.iloc[i].dropna().index)
 if len(z)>=8: turns.append(np.mean(abs(rank.iloc[i][z]-rank.iloc[i-10][z])))
print('coverage=%.4f turnover10=%.4f'%(f.notna().mean().mean(),np.mean(turns)))
for lab,idx in [('2020-23',p.index[p.index.year<=2023]),('2024-27',p.index[(p.index.year>=2024)&(p.index.year<=2027)]),('2028-30',p.index[(p.index.year>=2028)&(p.index.year<=2030)]),('2031+',p.index[p.index.year>=2031]),('recent120',p.index[-120:])]:
 n,ic,ir,nn=calc(1,idx); print(lab,'dates=%d IC=%.6f ICIR=%.6f meanN=%.2f'%(n,ic,ir,nn))
# decay by horizon on latest 120
for h in [1,5,10,20]: print('RECENT H%d'%h,calc(h,p.index[-120:])[:3])
