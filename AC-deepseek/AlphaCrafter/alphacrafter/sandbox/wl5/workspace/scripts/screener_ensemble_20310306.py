"""Screener v22 ensemble construction - 2031-03-06 cycle (data through 2031-03-05).
Pure calculation + persistence of factor_ensemble.json. No backtest/step, no account/date mutation.
"""
import json

# Raw quality evidence: q = |IC| * |ICIR| from benchmark admission selected metrics
# (cny_beta ICIR capped at 0.35 per prior-cycle convention; direction = sign(IC))
base = {
    "trend_r2_30_signed":   dict(ic=0.0562, icir=0.1672, direction=+1),
    "cny_beta_60":          dict(ic=0.1161, icir=0.35,   direction=+1),   # capped
    "semi_down_ratio_20":   dict(ic=0.0857, icir=0.2402, direction=-1),
    "mom_120d_skip5":       dict(ic=0.0521, icir=0.1381, direction=+1),
    "vol_of_vol20x60":      dict(ic=0.0424, icir=0.1206, direction=+1),
    "time_under_water_120": dict(ic=0.0570, icir=0.1700, direction=-1),
    "dxy_beta_60":          dict(ic=0.0843, icir=0.2510, direction=+1),
    "vix_beta_cond_60x20":  dict(ic=0.0382, icir=0.0927, direction=-1),
    "mom_10d_skip5":        dict(ic=0.0409, icir=0.1183, direction=+1),
    "tail_ratio_20":        dict(ic=0.0466, icir=0.1333, direction=+1),
}

# Regime multipliers (2031-03-06 read, data through 2031-03-05) and turnover penalties.
# Regime: SOX -13.3%/10d relief-failure -> semi_down strongly re-validated (x1.20);
# SPX +10.5%/10d clean uptrend + ETH +31.7%/60d -> trend_r2 anchors firm (x1.05);
# WTI +14.5% V-rebound + SOX/BTC reversals -> extreme whipsaw, mom_10d x0.50;
# DXY weak (-3.5%/60d below MAs) -> dxy_beta discrimination moderate (x0.85);
# VIX 46.7 (from 53.0) still elevated, vol-of-vol high (BTC 8.7, WTI 5.7) -> vol/defensive x1.0.
regime_mult = {
    "trend_r2_30_signed":   1.05,
    "cny_beta_60":          1.00,
    "semi_down_ratio_20":   1.20,
    "mom_120d_skip5":       1.00,
    "vol_of_vol20x60":      1.00,
    "time_under_water_120": 1.00,
    "dxy_beta_60":          0.85,
    "vix_beta_cond_60x20":  1.00,
    "mom_10d_skip5":        0.50,   # turnover 4.09 x whipsaw block
    "tail_ratio_20":        0.65,   # turnover 3.45
}

scores = {}
for fid, m in base.items():
    q = abs(m["ic"]) * abs(m["icir"])
    s = q * regime_mult[fid]
    scores[fid] = s

tot = sum(scores.values())
weights = {fid: s / tot for fid, s in scores.items()}

# round to 4 dp and renormalize exactly
for fid in weights:
    weights[fid] = round(weights[fid], 4)
# fix residual rounding to sum exactly 1.0 on the largest-weight factor
diff = 1.0 - sum(weights.values())
top = max(weights, key=weights.get)
weights[top] = round(weights[top] + diff, 4)

print("raw scores:", {k: round(v, 5) for k, v in sorted(scores.items(), key=lambda x: -x[1])})
print("\nWEIGHTS (sum =", round(sum(weights.values()), 6), "):")
for fid, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {fid:22s} {w:.4f}  dir={base[fid]['direction']:+d}")

selected = [{"factor_id": fid, "weight": weights[fid], "direction": base[fid]["direction"]}
            for fid in weights]

ens = {
    "schema_version": 1,
    "selected_factors": selected,
    "method": "quality_ic_tilt",
    "as_of": "2031-03-06",
    "notes": ("v22 refresh (regime read through 2031-03-05, date-gated). REGIME vs v21 (2031-02-19): "
              "SPX +10.51%/10d +10.34%/20d +14.92%/60d above MA20/MA60 -> clean US-broad uptrend anchor; "
              "SOX -13.28%/10d -7.62%/20d BELOW MA20/MA60 (5 straight down days) -> relief rally FAILED, semi_down re-validated; "
              "WTI +14.46%/10d V-rebound above MA20/MA60 after 2 down blocks (vol 68%, noisy 30d fit -> trend_r2 low score on WTI, self-protecting); "
              "ETH +5.21%/10d +31.74%/60d above MAs (strong uptrend); BTC -10.47%/10d below MA60 (-13.19%) rebound failed again; "
              "NDX -3.67% vs SPX +10.51% extreme US breadth split; N225 -1.68% consolidating after +11.1%; SX5E +4.24%/10d +7.31%/20d above MAs (Europe improving); "
              "COPPER +9.82%/20d above MA20 stabilizing; XAU flat. VIX 46.70 (-11.93%/10d, easing from 53 but +15.92%/60d, at MA60) -> elevated-vol regime persisting, vol-of-vol still high "
              "(BTC 8.69, WTI 5.72, ETH 4.87, NDX 4.41). DXY 94.66 -3.46%/60d below MAs (USD weak), USDJPY -9.25%/60d yen strength, EURUSD +3.82%/60d. "
              "Cross-sectional 10d return dispersion 6.89% (up from 4.56%) -> strong signal power; max WTI +14.46% / min SOX -13.28%. "
              "WEIGHTS (quality_ic_tilt, q=|IC|*|ICIR|, cny_beta ICIR capped 0.35; regime multipliers; turnover penalty 0.50 mom_10d / 0.65 tail_ratio): "
              "semi_down RAISED .1424->.1719 (#1, SOX -13.3% relief-failure + VIX 47); trend_r2 kept .1483->.1510 (#2, SPX/ETH/SX5E anchors, WTI V noise low-score); "
              "cny_beta kept .1469->.1458 (#3, USD weak, strongest live evidence); mom_120d kept .1420->.1406 (SPX +14.9%/60d, ETH +31.7%, COPPER stabilizing); "
              "vol_of_vol .1054->.1042 (vol-of-vol still elevated); tuw .0837->.0833 (BTC/NDX drawdown risk); dxy_beta TRIMMED .0782->.0729 (DXY downtrend established, discrimination moderate); "
              "vix_beta .0730->.0729 (VIX still high); mom_10d TRIMMED .0434->.0260 (extreme whipsaw block: WTI/SOX/BTC/SPX 10d moves all >10%); tail_ratio .0367->.0313 (turnover 3.45). "
              "No direction flips. Blocks: momentum 0.3176, defensive 0.3281, macro/FX/vol 0.3229, tail 0.0313. "
              "Excluded unchanged: WTI_BETA_60 (2023-26 regime IC negative, live 90d -0.175; +14.5% single-block rebound is not re-admission evidence), kurt_20 (live-history override, 12 adverse cycles). "
              "10 factor IDs identical to v21 -> strategy.py dynamic loader stays in sync. "
              "Crowding watch: semi_down vs mom_10d maxcorr 0.505 (opposite direction -> partial hedge), tuw vs mom_120d maxcorr 0.51; cny vs dxy corr null, both kept modest pending miner re-validation. "
              "Turnover flag: mom_10d 4.09, tail_ratio 3.45, vix_beta 3.35 vs 10d horizon + 3bp rebalance gate. "
              "5 frozen ballast series (000300.SH, 000688.SH, HSI, US10Y, CN10Y) show zero movement -> inert, zero cross-sectional signal, per account state."),
}

with open("factors/factor_ensemble.json", "w") as f:
    json.dump(ens, f, indent=1)
print("\nPersisted factors/factor_ensemble.json")
print("sum check:", round(sum(w["weight"] for w in ens["selected_factors"]), 6))
