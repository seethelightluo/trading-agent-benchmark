import json, os
# Check current active factor files for signal artifacts / structure
for f in ['mom_120d_skip5.json', 'vix_beta_cond_60x20.json', 'dxy_beta_60.json', 'time_under_water_120.json']:
    with open(f'factors/{f}') as fp:
        d = json.load(fp)
    keys = list(d.keys())
    has_sig = 'signal_artifacts' in d
    val = d.get('validation', {})
    print(f, '| keys:', keys)
    print('   status:', val.get('status'), '| last_validated:', val.get('last_validated'), '| period:', val.get('period'))
    print('   metrics ic/icir:', val.get('metrics', {}).get('ic'), val.get('metrics', {}).get('icir'), '| has_signal_artifacts:', has_sig)
    if has_sig:
        sa = d['signal_artifacts']
        print('   signal_artifacts keys:', list(sa.keys()) if isinstance(sa, dict) else type(sa))
    print()
