"""One idea: peer-tail transition residual (20d change in 20d upside vs downside event frequency)."""
import numpy as np
# Re-use the complete data, validation and admitted-library reconstruction framework
# from the immediately preceding candidate; replace only its factor construction.
source_path='scripts/miner_2_20280323_net_peer_tail_persistence_residual_60obs.py'
src=open(source_path,encoding='utf-8').read()
prefix=src.split('def metric(h):')[0]
# Prefix defines p, r, vol, trend and an obsolete f.  Rebuild f as a distinct
# transition (recent tail balance less prior tail balance), residualized daily.
exec(prefix,globals())
hi=r.quantile(.8,axis=1); lo=r.quantile(.2,axis=1)
up=r.ge(hi,axis=0).astype(float).where(r.notna())
dn=r.le(lo,axis=0).astype(float).where(r.notna())
net=up-dn
recent=net.rolling(20,min_periods=16).mean()
prior=recent.shift(20)
raw=recent-prior
# Remove persistent tail-state and conventional volatility-adjusted trend.
f=raw*np.nan
for d in p.index:
 z=__import__('pandas').concat([raw.loc[d].rename('y'),trend.loc[d].rename('t'),recent.loc[d].rename('s')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['t','s']]]
  f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
# The rest supplies IC/decay/regime/turnover and reconstructs all 16 admitted signals.
rest='def metric(h):'+src.split('def metric(h):',1)[1]
exec(rest,globals())
