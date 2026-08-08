"""One idea: VIX-elevated US10Y downside-shock beta resilience (60), fully audited."""
p='scripts/miner_2_20310612_us10y_downside_shock_beta_resilience_60_audit.py'
s=open(p,encoding='utf-8').read()
s=s.replace('"""One idea: US10Y downside-shock beta resilience (60), audited against admitted library."""','"""One idea: VIX-elevated US10Y downside-shock beta resilience (60), audited against admitted library."""')
old="""# Only unusually negative tradable-US10Y return days: easing/rates-down shock.
y10early=r['US10Y']
ystate=y10early<y10early.rolling(60,min_periods=40).quantile(.25)
cand=cs(-pd.DataFrame({a:eb0(r[a],y10early,ystate,60)-beta(r[a],y10early,60) for a in A})).shift(1)"""
new="""# Relative US10Y downside sensitivity only during independently elevated VIX levels.
# Both predicates are lagged by the final shift, so no same-day/future information enters a score.
y10early=r['US10Y']
vix_elevated=vix>vix.rolling(60,min_periods=40).median()
ystate=(y10early<y10early.rolling(60,min_periods=40).quantile(.25)) & vix_elevated
cand=cs(-pd.DataFrame({a:eb0(r[a],y10early,ystate,60)-beta(r[a],y10early,60) for a in A})).shift(1)"""
assert old in s
s=s.replace(old,new).replace("FACTOR us10y_downside_shock_beta_resilience_60 CUTOFF","FACTOR vix_elevated_us10y_downside_shock_beta_resilience_60 CUTOFF")
exec(compile(s,'<vix_elevated_us10y_downside_audit>','exec'))
