import json
p='factors/miner_3_20270208_breadth_confirmed_momentum.json'
obj={
 'factor_id':'miner_3_20270208_breadth_confirmed_momentum',
 'factor_name':'Breadth-Confirmed Volatility-Scaled Momentum',
 'version':'20270208',
 'calculation':{'expression':'shift((return_20d / realized_vol_20d) * (0.5 + abs(cross_asset_breadth_20d - 0.5)), 1)','description':'For each asset, scale 20-day momentum by its 20-day realized volatility and amplify the signal when the cross-asset fraction of positive daily returns is far from neutral. The one-day shift prevents look-ahead.'},
 'dependencies':['close','daily_returns'],
 'parameters':{'momentum_window':20,'volatility_window':20,'breadth_window':20,'breadth_min_periods':10,'vol_min_periods':12,'lag_days':1,'universe_size':15},
 'validation':{'status':'EFFECTIVE','period':{'start':'2020-01-01','end':'2027-02-07','data_cutoff':'2027-02-05'},'metrics':{'daily_ic':0.02095896408841766,'daily_icir':1.0296266214497347,'ic_hit_ratio':0.5252043596730245,'valid_ic_dates':1468,'average_valid_instruments':14.435967302452315,'factor_coverage':1.0,'turnover':None,'decay':{'1d':0.02095896408841766,'5d':0.022336063993306298,'10d':0.04874576544744087,'20d':0.049501600170211275},'max_abs_library_correlation':None},'regime_notes':{'2020-2022':{'ic':0.03733962931134312,'icir':1.878327309265539},'2023-2024':{'ic':-0.0036270030782225996,'icir':-0.1728590952167692},'2025-2027':{'ic':0.02147420859464151,'icir':1.054604502333563}},'validation_timestamp':'2027-02-08T00:00:00Z'},
 'signal_provenance':{'script':'scripts/miner_3_20270208_breadth_confirmed_momentum.py','signal_artifact':'scripts/miner_3_20270208_breadth_confirmed_momentum_signal.csv','signal_lag':'1 trading day'},
 'tags':['momentum','trend','volatility_scaled','cross_asset_regime']
}
with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2)
with open(p,encoding='utf-8') as f: x=json.load(f)
assert x['factor_id']=='miner_3_20270208_breadth_confirmed_momentum' and x['validation']['status']=='EFFECTIVE'
assert abs(x['validation']['metrics']['daily_ic'])>=.007 and abs(x['validation']['metrics']['daily_icir'])>=.084
assert x['signal_provenance']['signal_artifact']
print(json.dumps({'factor_id':x['factor_id'],'status':x['validation']['status'],'daily_ic':x['validation']['metrics']['daily_ic'],'daily_icir':x['validation']['metrics']['daily_icir']}))
