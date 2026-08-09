import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# One idea: VIX-up yield-shock peer-relative beta. Assets whose peer-relative returns
# have held up during simultaneous rising-volatility / large-yield-move sessions may lead.
def close(symbol, root):
 x=pd.read_csv(f'{root}/{symbol}.csv',usecols=['date','close']).rename(columns={'close':symbol}).set_index('date'); x.index=pd.to_datetime(x.index); return x
P=pd.concat([close(a,'../persistent/stock_data') for a in A],axis=1).sort_index()
y=close('US10Y','../persistent/stock_data').US10Y.reindex(P.index).pct_change()
v=close('VIX','../persistent/index_data').VIX.reindex(P.index).pct_change()
r=P.pct_change(); rel=r.sub(r.median(axis=1),axis=0)
# predetermined trailing thresholds only; interaction is continuous rather than event-picked
zy=(y-y.rolling(60,min_periods=40).mean())/y.rolling(60,min_periods=40).std()
zv=(v-v.rolling(60,min_periods=40).mean())/v.rolling(60,min_periods=40).std()
stress=(zy.abs()*zv.clip(lower=0)).clip(upper=5)
sig=rel.mul(stress,axis=0).rolling(60,min_periods=40).mean()
sig=sig.sub(sig.median(axis=1),axis=0).shift(1)
def ic_stats(h, mask=None):
 f=P.shift(-h).div(P)-1; S=sig if mask is None else sig.loc[mask]; F=f.reindex(S.index); out=[]; breadth=[]
 for d in S.index:
  ok=S.loc[d].notna() & F.loc[d].notna()
  if ok.sum()>=8: out.append(spearmanr(S.loc[d][ok],F.loc[d][ok]).statistic); breadth.append(ok.sum())
 q=np.array(out); return {'dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)),'hit':float((q>0).mean()),'breadth_mean':float(np.mean(breadth)),'breadth_min':int(np.min(breadth))}
print('CANDIDATE vix_up_yield_shock_peer_relative_beta_60')
print('cutoff',P.index.max().date(),'cells',int(sig.notna().sum().sum()),'/',sig.size,'coverage',float(sig.notna().mean().mean()))
for h in [1,5,10,20]: print('H',h,ic_stats(h))
for n,m in [('2023_2026',(sig.index>='2023-01-01')&(sig.index<'2027-01-01')),('2027_now',sig.index>='2027-01-01'),('recent180',sig.index>=sig.index.max()-pd.Timedelta(days=180))]: print('REGIME_10',n,ic_stats(10,m))
print('turnover',float(sig.rank(axis=1,pct=True).diff().abs().stack().mean()),'dispersion',float(sig.std(axis=1).mean()))
print('LIBRARY_CORRELATION_EVIDENCE unavailable: admitted-factor historical signal matrices cannot be reconstructed reliably from JSON definitions alone; max_abs_library_correlation is therefore missing and admission is prohibited.')
