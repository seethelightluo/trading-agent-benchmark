# Exact active-library novelty audit for beta-neutral residual trend acceleration.
# It extends the definition-driven reconstruction with every current factor omitted
# in the earlier template; candidate construction replaces that template's candidate.
p='scripts/miner_3_20320108_peer_relative_trend_acceleration_10_30_audit.py'
s=open(p).read()
s=s.replace('"""One idea: continuous VIX-directional relative-return asymmetry, with admitted-library novelty audit."""','"""Exact active-library novelty audit: beta-neutral residual trend acceleration."""')
s=s.replace("# Candidate: peer-relative 10-session return minus non-overlapping prior 30-session return.\nrecent=P.pct_change(10)\nprior=P.shift(10).pct_change(30)\ncand=cs(recent-prior).clip(-1,1).shift(1)","""# Candidate: 40d leave-one-out-peer-beta residual trend acceleration, lagged one day.
res=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 peer=other[a]
 b=r[a].rolling(40,min_periods=30).cov(peer)/peer.rolling(40,min_periods=30).var()
 res[a]=r[a]-b*peer
recent=res.rolling(10,min_periods=8).sum(); prior=res.shift(10).rolling(30,min_periods=23).sum()
rv=res.rolling(40,min_periods=30).std()*np.sqrt(40)
cand=cs((recent-prior)/rv).clip(-5,5).shift(1)""")
needle="# Repair event-magnitude construct that is intentionally common across date then needs cross-sectional rel."
extra="""# Current active definitions omitted by the legacy template.
# lag-5 peer-relative serial dependence (not the legacy lag-1 version).
S['inverse_peer_relative_lag5_serial_dependence_40']=cs(-pd.DataFrame({a:rel[a].rolling(40,min_periods=30).corr(rel[a].shift(5)) for a in A})).shift(1)
# Continuous and moderate broad-weakness peer capture.
peer_med=r.median(axis=1); q20=peer_med.rolling(60,min_periods=40).quantile(.20)
S['moderate_downside_peer_relative_capture_60']=cs(pd.DataFrame({a:(r[a]-peer_med).where((peer_med<0)&(peer_med>q20)).rolling(60,min_periods=12).mean() for a in A})).shift(1)
S['continuous_broad_weakness_relative_capture_60']=cs(pd.DataFrame({a:(r[a]-peer_med).where(peer_med<0).rolling(60,min_periods=15).mean() for a in A})).shift(1)
# Magnitude-weighted extreme weakness capture with leave-one-out peers.
ex={}
for a in A:
 peer=other[a]; q=peer.rolling(60,min_periods=40).quantile(.25).shift(1); w=(-peer).where(peer<q)
 ex[a]=-((r[a]-peer)*w).rolling(60,min_periods=12).sum()/w.rolling(60,min_periods=12).sum()
S['inverse_extreme_broad_weakness_magnitude_weighted_relative_capture_60']=cs(pd.DataFrame(ex)).shift(1)
# Five-day broad-stress-onset peer reversal.
m5=P.pct_change(5).median(axis=1); stress=m5<=m5.rolling(60,min_periods=40).quantile(.20)
S['broad_stress_onset_peer_reversal_5_60']=cs((-P.pct_change(5).add(m5,axis=0)).where(stress,axis=0)).shift(1)
"""
s=s.replace(needle,extra+'\n'+needle).replace('FACTOR peer_relative_trend_acceleration_10_30','FACTOR beta_neutral_residual_trend_acceleration_10_40')
open('scripts/miner_1_20320304_beta_neutral_residual_trend_acceleration_10_40_exact_audit.py','w').write(s)
print('written')
