import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
px={}
for s in syms:
    d=pd.read_csv(base/f'{s}.csv')
    d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
    px[s]=d['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill()
r=np.log(p).diff()
# Residual relative-strength momentum: medium horizon excess return versus cross-asset median,
# risk-normalized, lagged one day. Tests whether persistent leaders outperform peers.
excess=r.rolling(40).sum().sub(r.rolling(40).sum().median(axis=1),axis=0)
vol=r.rolling(40).std()*np.sqrt(252)
f=(excess/vol).shift(1)
rows=[]
for h in [5,10,20]:
    fw=np.log(p.shift(-h)/p)
    vals=[]
    for dt in f.index:
        a=f.loc[dt]; b=fw.loc[dt]
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
    x=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
    print(f'H={h} dates={len(x)} avg_n={x.n.mean():.2f} IC={x.ic.mean():.6f} ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f} hit={(x.ic>0).mean():.4f}')
    for label,cut in [('2026+', '2026-01-01'),('2027+','2027-01-01'),('2028+','2028-01-01')]:
        y=x[x.index>=cut]
        print(f'  {label} n={len(y)} IC={y.ic.mean():.6f} ICIR={y.ic.mean()/y.ic.std(ddof=1):.6f}')
# rank turnover / coverage
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().mean(axis=1).dropna().mean()
print(f'panel_dates={len(f)} instruments={len(syms)} coverage={f.notna().mean().mean():.6f} rank_turnover={turn:.6f}')
out=f.copy(); out.index.name='date'; out.reset_index().to_csv('scripts/miner_1_20280810_residual_relative_strength_signal.csv',index=False)
