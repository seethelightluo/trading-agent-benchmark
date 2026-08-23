import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2029-06-27')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in syms:
    p='../persistent/stock_data/'+s+'.csv'
    if not os.path.exists(p): p='../persistent/index_data/'+s.replace('.','_')+'.csv'
    if os.path.exists(p):
        d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
        P[s]=d['close'].astype(float)
px=pd.DataFrame(P).sort_index().loc[:cut]
# causal factor: 60d return normalized by trailing 20d realized volatility
ret=px.pct_change()
f=(px/px.shift(60)-1)/(ret.rolling(20,min_periods=15).std()*np.sqrt(20))
# future 10 observations aligned at t
fw=px.shift(-10)/px-1
rows=[]; sig=[]
for dt in f.index:
    a=f.loc[dt]; b=fw.loc[dt]; ok=a.notna()&b.notna()
    if ok.sum()>=8:
        ic=spearmanr(a[ok],b[ok]).statistic
        if np.isfinite(ic): rows.append((dt,ic,ok.sum()))
    for s in syms:
        if pd.notna(f.loc[dt,s]): sig.append((dt,s,float(f.loc[dt,s])))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('cutoff',cut.date(),'instruments',len(P),'panel_rows',len(px),'IC_dates',len(r),'avg_n',r.n.mean())
print('mean_ic %.6f icir %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for lo,hi in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-06-27')]:
 x=r.loc[lo:hi].ic; print(lo,hi,len(x), 'ic %.6f icir %.6f'%(x.mean(),x.mean()/x.std(ddof=1)) if len(x)>2 else 'NA')
# turnover of rank ordering / normalized score, daily available observations
S=pd.DataFrame([(d,s,v) for d,s,v in sig],columns=['date','symbol','v']).pivot(index='date',columns='symbol',values='v')
rank=S.rank(pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna().mean()
print('coverage %.4f turnover %.4f'%(S.notna().mean().mean(),turn))
out='scripts/miner_2_20290628_voladj_trend_60d_signal.csv'; S.stack().rename('signal').reset_index().to_csv(out,index=False); print('artifact',out)
