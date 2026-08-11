import json

ids = ["rel_mom_20d_skip5", "vix_beta_cond_60x20", "vol_adj_mom_20x60",
       "downside_vol_ratio_20", "mom_120d_skip5", "vol_of_vol20x60",
       "eurusd_beta_cond_60x20", "beta_ew_60d", "amihud_20"]
for fid in ids:
    d = json.load(open(f"factors/{fid}.json"))
    print("=" * 20, fid)
    print("name:", d.get("factor_name"))
    print("calc:", json.dumps(d.get("calculation", {}), indent=1))
    print("params:", d.get("parameters"))
    print("direction:", d.get("expected_direction"))
    m = d.get("validation", {}).get("metrics", {})
    print("ic:", m.get("ic"), "icir:", m.get("icir"), "turnover_10d_rank:", m.get("turnover_10d_rank"))
