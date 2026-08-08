import json, os
src='factors/miner_2_20260716_realized_volatility_20obs.json'
dst='factors/miner_2_20260716_realized_volatility_20obs_deprecated.json'
with open(src,encoding='utf-8') as h: x=json.load(h)
x['version']='2026-08-27'
x['last_validated']='2026-08-27'
v=x['validation'];v['period']='2020-01-01 to 2026-08-26';v['timestamp']='2026-08-27';v['status']='DEPRECATED'
m=v['metrics'];m.update({'daily_paper_ic':0.0254340122,'daily_paper_icir':0.0744504391,'ic_std':0.3416234006,'ic_hit_ratio':0.5204021289,'ic_dates':1691,'mean_valid_instruments_per_ic_date':14.5452395033,'signal_cell_coverage':0.7239867659,'mean_valid_instruments_per_signal_date':10.8598014888,'mean_rank_turnover_10d':0.1012136916,'decay_ic':{'1d':0.0126769254,'5d':0.02448484,'10d':0.0254340122,'20d':0.0221506202},'decay_icir':{'1d':0.0359086861,'5d':0.0700679834,'10d':0.0744504391,'20d':0.0671088333}})
v['regime_notes']={'2020':{'dates':244,'ic':0.1137518219,'icir':0.3012432971},'2021_2022':{'dates':516,'ic':0.0432081323,'icir':0.1221124157},'2023_2024':{'dates':516,'ic':0.0118843527,'icir':0.0394620008},'2025_2026_08_26':{'dates':415,'ic':-0.0317451861,'icir':-0.0933476274},'latest_90_calendar_days':{'dates':54,'ic':-0.145818869,'icir':-0.6142061997},'interpretation':'Full-sample 10d ICIR fell below 0.084 and latest 90 calendar days are strongly negative; deprecated.'}
x['benchmark_admission']['deprecation_reason']='2026-08-27: primary 10d ICIR 0.074450 < 0.084000; latest regime IC/ICIR negative.'
with open(dst,'w',encoding='utf-8') as h: json.dump(x,h,indent=2)
os.remove(src)
print('DEPRECATED',dst)
