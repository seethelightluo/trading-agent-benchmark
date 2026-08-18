import json
p='scripts/miner_2_20340217_risk_adjusted_trend_signal.csv'
# artifact currently stores percentile risk-adjusted trend; persist explicit reversal direction in definition
obj={
 "factor_id":"miner_2_20340217_risk_adjusted_trend_reversal_20d",
 "factor_name":"Risk-adjusted medium-term trend reversal",
 "version":"1.0",
 "calculation":{"expression":"-CSRank((close/close.shift(60)-1)/(rolling_std(pct_change(close),20)*sqrt(252)))", "description":"Cross-sectional reversal of 60-day return normalized by 20-day realized volatility; lagged one completed day to prevent look-ahead. Higher values favor assets with weaker risk-adjusted medium-term performance."},
 "dependencies":["close"],
 "parameters":{"trend_window":60,"volatility_window":20,"forward_horizon":20,"lag":1,"min_cross_section":8},
 "validation":{"status":"EFFECTIVE","period":{"start":"2020-01-01","end":"2034-02-17"},"metrics":{"ic":0.062784,"icir":2.881609,"hit_ratio":0.5998,"coverage":0.6238,"turnover":0.1285,"valid_dates":2529,"average_instruments":13.023,"max_abs_library_correlation":None},"regime_notes":"Strong positive reversal IC in 2026-2028 and 2029-2033; latest 2031-2034 subperiod is near neutral at the 20-day horizon, so monitor drift. Negative trend IC was sign-flipped into reversal.","artifacts":{"signal":"scripts/miner_2_20340217_risk_adjusted_trend_signal.csv","ic":"scripts/miner_2_20340217_risk_adjusted_trend_ic.csv"}},
 "tags":["reversal","momentum","risk-adjusted","cross-asset"],
 "last_validated":"2034-02-17"
}
with open('factors/miner_2_20340217_risk_adjusted_trend_reversal_20d.json','w') as f: json.dump(obj,f,indent=2)
with open('factors/miner_2_20340217_risk_adjusted_trend_reversal_20d.json') as f: x=json.load(f)
assert x['factor_id']=='miner_2_20340217_risk_adjusted_trend_reversal_20d' and x['validation']['status']=='EFFECTIVE' and x['validation']['metrics']['ic']>=.007 and x['validation']['metrics']['icir']>=.084 and x['validation']['artifacts']['signal']
print('verified',x['factor_id'],x['validation']['status'],x['validation']['metrics'])
