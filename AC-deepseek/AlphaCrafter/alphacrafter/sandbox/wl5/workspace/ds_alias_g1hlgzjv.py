
import json, os
# Check factor statuses and key metrics
for fid in ['trend_r2_30_signed','semi_down_ratio_20','mom_120d_skip5','mom_10d_skip5','time_under_water_120','vol_of_vol20x60','dxy_beta_60','WTI_BETA_60','vix_beta_cond_60x20','tail_ratio_20','kurt_20']:
    p = f'factors/{fid}.json'
    if os.path.exists(p):
        d = json.load(open(p))
        v = d.get('validation', {})
        m = v.get('metrics', {})
        print(f"{fid}: status={v.get('status')} ic={m.get('ic')} icir={m.get('icir')} last_validated={v.get('last_validated')}")
    else:
        print(f"{fid}: MISSING")
