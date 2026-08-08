"""Apply miner_2's 2033-07-21 scheduled validation evidence to its factor record."""
import json
from pathlib import Path
p=Path('factors/miner_2_20330428_inverse_dispersion_amplified_us_cn_rate_spread_transmission_residual_30.json')
d=json.loads(p.read_text())
v=d['validation']; m=v['metrics']
v['period']='Point-in-time query cutoff 2033-07-20; locally available completed source history produced 861 eligible forward-IC dates; 15-instrument benchmark universe'
v['status']='EFFECTIVE'
m.update({'ic':0.076020,'icir':0.272898,'ic_horizon_days':20,'ic_dates':861,'hit_ratio':0.5958,'mean_instruments':12.99,'turnover_mean_daily_rank':0.116019,'coverage':0.176750,'valid_factor_cells':11183,'concentration':'Conditional transmission signal with 11,183 valid cells and a mean 12.99 valid instruments per each of 861 IC dates; every IC date meets the required eight-instrument minimum.','max_abs_library_correlation':0.131306,'closest_library_factor':'oil_shock_asymmetry','closest_common_valid_cells':10650})
m['decay']={'1_day':{'ic':0.014674,'icir':0.049989,'dates':861,'hit_ratio':0.5099},'5_day':{'ic':0.027752,'icir':0.093927,'dates':861,'hit_ratio':0.5459},'10_day':{'ic':0.046032,'icir':0.152912,'dates':861,'hit_ratio':0.5610},'20_day':{'ic':0.076020,'icir':0.272898,'dates':861,'hit_ratio':0.5958}}
v['regime_notes']='At the 20-session selected horizon, stable positive evidence holds in both partitions: 2026-2027 (349 dates), IC +0.089345, ICIR +0.285553, hit 57.02%; 2028 through available completed history (512 dates), IC +0.066937, ICIR +0.265165, hit 61.33%. The factor clears shared IC/ICIR gates at 5, 10 and 20 sessions. Reconstructed 25-signal library screen has maximum pooled Spearman |rho| 0.131306 against oil_shock_asymmetry over 10,650 common valid cells, below 0.5000. Query cutoff was 2033-07-20; completed local source history did not add observations beyond the prior validation, so this confirms stability but does not create fresher realized-return evidence.'
d['last_validated']='2033-07-21T00:00:00Z';d['revalidation_due']='2033-10-21'
d['benchmark_admission']['selected_metrics'].update({'ic':0.076020,'icir':0.272898,'max_abs_library_correlation':0.131306,'quality':0.02074570596})
p.write_text(json.dumps(d,indent=2)+'\n')
print('updated',p,'status',v['status'],'last_validated',d['last_validated'])
