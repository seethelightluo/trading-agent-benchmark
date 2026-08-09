import json
from pathlib import Path

def update(path, version, period, last, due, coverage, notes):
    p=Path(path); x=json.loads(p.read_text())
    x['version']=version
    x['validation']['period']=period
    x['validation']['metrics']['coverage']=coverage
    x['validation']['regime_notes']=notes
    x['last_validated']=last
    x['revalidation_due']=due
    p.write_text(json.dumps(x,indent=2)+"\n")

update('factors/miner_2_20330428_inverse_dispersion_amplified_us_cn_rate_spread_transmission_residual_30.json',
 '2033-10-13 revalidation',
 'Point-in-time query cutoff 2033-10-12; locally available completed source history produced 861 eligible forward-IC dates; 15-instrument benchmark universe',
 '2033-10-13T00:00:00Z','2034-01-13',0.174271,
 'Scheduled fixed-specification revalidation executed 2033-10-13 through prior completed session 2033-10-12. No future data used. The locally available completed history supplies 861 eligible IC dates with mean 12.99 instruments, each meeting the eight-name minimum. Selected 20d IC/ICIR is +0.076020/+0.272898, hit 59.58%; decay is 1d +0.014674/+0.049989, 5d +0.027752/+0.093927, 10d +0.046032/+0.152912, and 20d +0.076020/+0.272898. Regime 2026-27: 349 dates, +0.089345/+0.285553, 57.02% hit; 2028-current: 512 dates, +0.066937/+0.265165, 61.33% hit. Full 25-signal admitted-library screen: maximum absolute pooled Spearman correlation 0.131306 versus oil_shock_asymmetry on 10,650 common cells. All binding gates pass; retained EFFECTIVE.')

update('factors/miner_2_20310403_inverse_equity_stress_amplified_rate_transmission_residual_30.json',
 '2033-10-13 revalidation',
 'Point-in-time query cutoff 2033-10-12; locally available completed source history produced 878 eligible forward-IC dates; 15-instrument benchmark universe',
 '2033-10-13T00:00:00Z','2034-01-13',0.177871,
 'Scheduled fixed-specification revalidation executed 2033-10-13 through prior completed session 2033-10-12. No future data used. The local history supplies 878 eligible IC dates with mean 13.00 instruments, each meeting the eight-name minimum. Selected 20d IC/ICIR is +0.057478/+0.195932, hit 58.43%; decay is 1d +0.019868/+0.066564, 5d +0.027184/+0.085461, 10d +0.038779/+0.126956, and 20d +0.057478/+0.195932. Regime 2026-27: 366 dates, +0.062635/+0.184151, 57.92% hit; 2028-current: 512 dates, +0.053791/+0.210940, 58.79% hit. Full 25-signal admitted-library screen: maximum absolute pooled Spearman correlation 0.128667 versus dxy_shock_asymmetry on 10,813 common cells. All binding gates pass; retained EFFECTIVE.')
