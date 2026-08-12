import json

# Current 10-factor defensive set (kept for regime continuity; validated 2020-2026 warm-up, live since 2028-06)
factors = {
    'cn10y_beta_60':       {'ic': -0.0622, 'icir': -0.1871, 'regime_mult': 1.20},  # defensive rates beta
    'vol_adj_mom_20_60':   {'ic':  0.0582, 'icir':  0.1746, 'regime_mult': 0.90},  # high turnover 3.34, whipsaw
    'hs300_beta_60':       {'ic': -0.0449, 'icir': -0.1253, 'regime_mult': 1.10},  # defensive CN beta
    'comm_basket_beta_60': {'ic':  0.0428, 'icir':  0.1219, 'regime_mult': 0.80},  # commodity pullback flag
    'hilo_vol_ratio_20':   {'ic':  0.0418, 'icir':  0.1290, 'regime_mult': 0.90},  # turnover 4.0 highest
    'vol_of_vol20x60':     {'ic':  0.0424, 'icir':  0.1206, 'regime_mult': 1.00},
    'vix_beta_cond_60x20': {'ic': -0.0382, 'icir': -0.0927, 'regime_mult': 1.50},  # VIX 73 extreme stress -> boost
    'vol_regime_switch_20x60': {'ic': 0.0375, 'icir': 0.1313, 'regime_mult': 1.10},
    'intraday_ret_skew_20':{'ic':  0.0395, 'icir':  0.1329, 'regime_mult': 1.00},
    'dd_duration_120_resid':{'ic': -0.0330,'icir': -0.1116, 'regime_mult': 1.20},  # defensive drawdown
}
direction = {
    'cn10y_beta_60': -1, 'vol_adj_mom_20_60': 1, 'hs300_beta_60': -1, 'comm_basket_beta_60': 1,
    'hilo_vol_ratio_20': 1, 'vol_of_vol20x60': 1, 'vix_beta_cond_60x20': -1,
    'vol_regime_switch_20x60': 1, 'intraday_ret_skew_20': 1, 'dd_duration_120_resid': -1,
}

q = {fid: abs(v['ic'])*abs(v['icir'])*v['regime_mult'] for fid, v in factors.items()}
total = sum(q.values())
weights = {fid: q[fid]/total for fid in q}

# Round to 4 dp and force sum=1
w = {fid: round(wt, 4) for fid, wt in weights.items()}
diff = 1.0 - sum(w.values())
# apply residual to largest weight
big = max(w, key=w.get)
w[big] = round(w[big] + diff, 4)

print("q-scores:", {k: round(v,5) for k,v in q.items()})
print("weights:", w)
print("sum:", sum(w.values()))

sel = [{"factor_id": fid, "weight": w[fid], "direction": direction[fid]} for fid in q]
ens = {"schema_version": 1, "selected_factors": sel, "method": "quality_ic_tilt"}
with open('factors/factor_ensemble.json','w') as f:
    json.dump(ens, f, indent=2)
print("written factors/factor_ensemble.json")
