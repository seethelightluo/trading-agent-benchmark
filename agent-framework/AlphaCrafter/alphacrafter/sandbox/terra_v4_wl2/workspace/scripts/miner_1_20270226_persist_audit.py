import json
p='factors/miner_1_20270225_volconfirm_reversal5.json'
x=json.load(open(p)); x['validation']['metrics']['max_abs_library_correlation']=0.7150259639
x['validation']['metrics']['correlation_audit']='2027-02-26 deterministic pairwise artifact audit; rho vs vix-dispersion=0.67014, tail-rebound=0.71503'
open(p,'w').write(json.dumps(x,indent=2))
with open('memory.txt','a') as f:
 f.write('\n20270226 miner_1 deterministic signal audit: volconfirm_reversal5 rho=0.67014 with VIX-dispersion reversal and rho=0.71503 with tail rebound (488 and 2669 overlapping asset-date observations respectively); VIX-dispersion vs tail rebound rho=0.82815 (430 observations). Maximum observed pairwise rho=0.71503 for the newly audited vol5 factor. Updated factors/miner_1_20270225_volconfirm_reversal5.json max_abs_library_correlation and audit provenance. Sparse artifact overlap means interpret conservatively; no new factor admitted this audit cycle.\n')
print(json.load(open(p))['factor_id'],json.load(open(p))['validation']['status'],json.load(open(p))['validation']['metrics']['max_abs_library_correlation'])
