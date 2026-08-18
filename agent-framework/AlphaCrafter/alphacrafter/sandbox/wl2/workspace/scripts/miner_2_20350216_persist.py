import json,glob,os
p='factors/miner_2_20350216_vol_scaled_momentum_10d.json'
obj={'factor_id':'miner_2_20350216_vol_scaled_momentum_10d','factor_name':'Volatility-scaled 10-day momentum','version':'2035-02-16','calculation':{'expression':'sum_return(close,10).shift(1) / (std(pct_change(close),40).shift(1)*sqrt(252))','description':'Prior 10-session return divided by lagged 40-session annualized total volatility; signals are cross-sectionally demeaned after 5/95% winsorization.'},'dependencies':['close'],'parameters':{'momentum_window':10,'volatility_window':40,'winsorize':[0.05,0.95],'forward_horizons':[5,10,20,40]},'validation':{'status':'EFFECTIVE','metrics':{'ic':0.022358,'icir':0.275,'icir_definition':'mean daily paper IC / standard deviation of daily paper IC','coverage':0.99088,'turnover':0.10873,'average_instruments':14.9989,'dates':4546,'instruments':15,'max_abs_library_correlation':None,'decay':{'5d':0.013046,'10d':0.022358,'20d':0.019911,'40d':0.008868}},'period':'2020-05-13 to 2035-02-15','regime_notes':'Strong 2020-2025; weaker 2026-2030; negative 2031-2035 recent regime. Interpret uncertainty conservatively due to 15-name cross-section.','signal_artifact':'not generated; deterministic recomputation required'},'tags':['momentum','volatility','cross-asset'],'last_validated':'2035-02-16T00:00:00Z'}
with open(p,'w') as f: json.dump(obj,f,indent=2)
print(json.load(open(p)))
PY
python - <<'PY'
import json
p='factors/miner_2_20350216_vol_scaled_momentum_10d.json'; x=json.load(open(p)); print(x['factor_id'],x['validation']['status'],x['validation']['metrics']['ic'],x['validation']['metrics']['icir'])
PY