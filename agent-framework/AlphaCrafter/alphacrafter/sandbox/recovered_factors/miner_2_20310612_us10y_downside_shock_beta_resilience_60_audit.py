"""One idea: US10Y downside-shock beta resilience (60), audited against admitted library."""
# Reuse the maintained complete validation/audit harness, replacing only its candidate.
p='scripts/miner_3_20310515_vix_volatility_transition_beta_resilience_60_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace('"""One idea: VIX volatility-transition beta resilience (60), with admitted-library novelty audit."""','"""One idea: US10Y downside-shock beta resilience (60), with admitted-library novelty audit."""')
old="""vstate=vr.abs().rolling(20,min_periods=15).mean()>vr.abs().rolling(60,min_periods=40).mean()
cand=cs(-pd.DataFrame({a:eb0(r[a],vr,vstate,60)-beta(r[a],vr,60) for a in A})).shift(1)"""
new="""# Only unusually negative US10Y return days: an interpretable easing/rates-down shock.
y10early=ix('US10Y').pct_change()
ystate=y10early<y10early.rolling(60,min_periods=40).quantile(.25)
cand=cs(-pd.DataFrame({a:eb0(r[a],y10early,ystate,60)-beta(r[a],y10early,60) for a in A})).shift(1)"""
assert old in s
s=s.replace(old,new).replace("print('FACTOR vix_volatility_transition_beta_resilience_60 CUTOFF'","print('FACTOR us10y_downside_shock_beta_resilience_60 CUTOFF'")
exec(compile(s,'<us10y_downside_shock_audit>','exec'))
