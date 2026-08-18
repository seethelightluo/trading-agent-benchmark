import json
p='factors/miner_1_20350316_short_medium_relative_impulse_10d.json'
with open(p) as f:x=json.load(f)
x['factor_name']='Medium trend minus short-term pullback'
x['calculation']['expression']='lag1(ROC(close,60) - ROC(close,5))'
x['calculation']['description']='Cross-asset ranking of 60-day return less recent 5-day return; lagged one completed day. Positive values favor established medium-term trend with a recent pullback, and the validated factor is used with this exact sign.'
with open(p,'w') as f:json.dump(x,f,indent=2)
with open(p) as f:y=json.load(f)
print(y['factor_id'],y['calculation']['expression'],y['validation']['status'],y['validation']['metrics']['ic'])
