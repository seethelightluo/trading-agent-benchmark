"""Current visible-data revalidation: yield-volatility-transition beta resilience only."""
# Starts from the original single-idea implementation and replaces its proxy library
# audit with the 28 other active persisted signals.
p='scripts/miner_1_20300404_yield_volatility_transition_beta_resilience_60.py'
s=open(p,encoding='utf-8').read()
# delete three non-persisted proxy signals from audit
for key in ["'conditional_downside_participation_avoidance_60':cs(downpart),", "'conditional_peer_upside_participation_60':cs(up),", "'commonality_expansion_transition_40':cs(corr20.rolling(20,min_periods=15).mean()-corr20.shift(20).rolling(20,min_periods=15).mean()),"]:
    s=s.replace(key,'')
# Source USDJPY once (observation-only macro input); append exact active-factor proxies.
s=s.replace("vix=ix('VIX');dxy=ix('DXY');vr=vix.pct_change();dr=dxy.pct_change();y10=r['US10Y']", "vix=ix('VIX');dxy=ix('DXY');usdjpy=ix('USDJPY');vr=vix.pct_change();dr=dxy.pct_change();ujr=usdjpy.pct_change();y10=r['US10Y']")
needle="'volscaled_reversal_1obs':-r/vol20}"
replacement="""'volscaled_reversal_1obs':-r/vol20,
# active miner_2 stress-duration reversal: inverse stress-weighted peer-relative return
'stress_duration_weighted_peer_resilience_reversal_60':cs((-rel.mul((1+.25*((m.lt(-.35*m.rolling(60,min_periods=40).std().shift(1))).astype(float).rolling(5,min_periods=1).sum().shift(1))).where(m.lt(-.35*m.rolling(60,min_periods=40).std().shift(1)),0.),axis=0).rolling(60,min_periods=25).sum().div(((1+.25*((m.lt(-.35*m.rolling(60,min_periods=40).std().shift(1))).astype(float).rolling(5,min_periods=1).sum().shift(1))).where(m.lt(-.35*m.rolling(60,min_periods=40).std().shift(1)),0.)).rolling(60,min_periods=25).sum(),axis=0)).shift(1)),
# active USDJPY transition beta resilience
'usdjpy_volatility_transition_beta_resilience_60':cs(-pd.DataFrame({a:eventbeta(r[a],ujr,((ujr.rolling(10,min_periods=8).std()>ujr.rolling(10,min_periods=8).std().rolling(60,min_periods=40).quantile(.75))&(ujr.rolling(10,min_periods=8).std()>ujr.rolling(10,min_periods=8).std().shift(5)),60) for a in A})+pd.DataFrame({a:beta(r[a],ujr,60) for a in A})).shift(1)}"""
assert needle in s
s=s.replace(needle,replacement)
s=s.replace("print('FACTOR yield_volatility_transition_beta_resilience_60", "print('REVALIDATION factor yield_volatility_transition_beta_resilience_60")
open('scripts/miner_1_20300613_revalidate_yield_volatility_transition.py','w',encoding='utf-8').write(s)
print('wrote current revalidation script')
