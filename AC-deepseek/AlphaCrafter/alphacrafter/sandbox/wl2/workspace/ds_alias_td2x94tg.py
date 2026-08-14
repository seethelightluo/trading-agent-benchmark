import json
for fn in ['factors/max_consec_gain_20.json','factors/mom_180d_skip5.json','factors/range_pos_252.json']:
    with open(fn) as f:
        d = json.load(f)
    print("====", fn)
    v = d.get('validation', {})
    m = v.get('metrics', {})
    print("ic:", m.get('ic'), "icir:", m.get('icir'), "n_ic_dates:", m.get('n_ic_dates'), "cov:", m.get('coverage_asset_days'))
    print("decay:", m.get('decay_ic_by_horizon'))
    print("max_abs_lib_corr:", m.get('max_abs_library_correlation'))
    ba = d.get('benchmark_admission', {})
    print("admission keys:", list(ba.keys()))
    print("artifact_provenance:", d.get('artifact_provenance'))