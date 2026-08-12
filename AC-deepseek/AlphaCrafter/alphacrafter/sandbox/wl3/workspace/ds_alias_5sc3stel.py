import json, glob
for f in ['gain_loss_asym_20','skew_term_20_60','idio_vol_ratio_60','vol_price_corr_60','updown_range_asym_20','max_ret_20d','win_rate_40','vol_shock_20']:
    try:
        d = json.load(open(f'factors/evicted/{f}.json.reason.json'))
        print(f, '->', str(d)[:220])
    except Exception as e:
        print(f, 'ERR', e)