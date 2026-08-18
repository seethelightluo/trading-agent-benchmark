import json,datetime
x={
 'factor_id':'miner_1_20350316_short_medium_relative_impulse_10d',
 'factor_name':'Short-versus-medium relative impulse', 'version':'1.0',
 'calculation':{'expression':'lag1(ROC(close,5) - ROC(close,60))','description':'Cross-asset ranking of recent 5-day return relative to 60-day return; lagged one completed day to avoid look-ahead. Positive values favor recent impulse, negative values favor medium-trend persistence.'},
 'dependencies':['close'], 'parameters':{'short_window':5,'medium_window':60,'signal_lag_days':1,'forward_horizon_days':10},
 'validation':{'status':'EFFECTIVE','last_validated':'2035-03-16T00:00:00Z','period':'2020-01-02 through 2035-03-16','metrics':{'ic':0.0660549981,'icir':0.1740843618,'hit_ratio':0.5753047582,'coverage':0.9846113514,'average_instruments':14.76917027,'valid_dates':2543,'total_instruments':15,'rank_turnover':0.0599789648,'decay_ic_5d':0.0458659828,'decay_icir_5d':0.12042239,'decay_ic_10d':0.0660549981,'decay_icir_10d':0.1741186,'decay_ic_20d':0.0748785627,'decay_icir_20d':0.19045458,'decay_ic_40d':0.1330800226,'decay_icir_40d':0.3520874,'max_abs_library_correlation':None},'regime_notes':'Inverse of tested trend-minus-short pullback signal. Positive 10-day IC in 2020-23 (0.0034), 2024-26 (0.0972), 2027-29 (0.0469), 2030-32 (0.1075), and 2033-35 (0.0456). Small cross-section requires conservative interpretation; artifact files provide recoverable signal and IC provenance.'},
 'signal_artifact':'scripts/miner_1_20350316_trend_pullback_signal.csv','ic_artifact':'scripts/miner_1_20350316_trend_pullback_ic.csv','tags':['cross_asset','momentum','relative_impulse','short_term']}
with open('factors/miner_1_20350316_short_medium_relative_impulse_10d.json','w') as f: json.dump(x,f,indent=2)
with open('factors/miner_1_20350316_short_medium_relative_impulse_10d.json') as f: y=json.load(f)
print(y['factor_id'],y['validation']['status'],y['validation']['metrics']['ic'],y['validation']['metrics']['icir'],y['signal_artifact'])
