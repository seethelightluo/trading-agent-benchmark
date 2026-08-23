"""miner_3 exploration 2031-08-20: novel cross-asset candidates through visible date."""
import sys, json
sys.path.insert(0, "scripts")
from miner_shared import (ASSETS, MACRO, load_close, load_macro, forward_ret, daily_ic,
                          ic_stats, rank_turnover, coverage_stats, library_panel, max_lib_corr)

END = "2031-08-20"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()

lib_panels = library_panel(close, macro)

# --- candidate constructors (return DataFrame indexed like close) ---
cands = {}

# 1) USDJPY-conditional beta (risk-on JPY weakening proxy)
dxy = macro["DXY"]; usdjpy = macro["USDJPY"]; eurusd = macro["EURUSD"]; vix = macro["VIX"]
r_usdjpy = usdjpy.pct_change(); usdjpy_20 = usdjpy / usdjpy.shift(20) - 1.0
jp_beta = ret.rolling(60, min_periods=30).cov(r_usdjpy) / r_usdjpy.rolling(60, min_periods=30).var()
cands["jpy_beta_cond_60x20"] = (-jp_beta.multiply(usdjpy_20, axis=0))

# 2) Vol reversal 5x60: recent vol relative to long vol (short vol spikes)
rv5 = ret.rolling(5).std(); rv60 = ret.rolling(60).std()
cands["vol_reversal_5x60"] = (-(rv5 / rv60))

# 3) Drawdown depth from 60d high (mean reversion long)
cands["dd_60d"] = (close / close.rolling(60).max() - 1.0)

# 4) Risk-adjusted momentum 40d (sharpe-like)
mom40 = close / close.shift(40) - 1.0
cands["risk_adj_mom_40"] = (mom40 / rv60.abs())

# 5) Range position 20d (already in explore; retest)
hi = close.rolling(20).max(); lo = close.rolling(20).min()
cands["range_pos_20"] = ((close - lo) / (hi - lo))

# 6) Aggregate cross-momentum: individual relative momentum vs mean of all
rmom = close / close.shift(60) - 1.0
rmom_rel = rmom.subtract(rmom.median(axis=1), axis=0)
cands["cross_mom_60_rel"] = rmom_rel

# 7) Downside-vol ratio longer horizon (40d daily-downside)
neg = ret.where(ret < 0, 0.0)
ds40 = (neg ** 2).rolling(40).mean().apply(lambda x: x ** 0.5)
cands["downside_vol_ratio_40"] = (-(ds40 / ret.rolling(40).std()))

fwd = forward_ret(close, 10)
print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}\n")
print(f"{'factor':<26}{'IC10':>8}{'ICIR':>8}{'hit':>6}{'n':>6} | {'IC_q':>8}{'ICIR_q':>8} | {'turn':>6} {'covD8':>6} | maxcorr")
for name in cands:
    f = cands[name]
    ic_s = daily_ic(f, fwd)
    st = ic_stats(ic_s, 10)
    f_q = f.tail(250)
    st_q = ic_stats(daily_ic(f_q, forward_ret(close, 10).reindex(f_q.index)), 10)
    turn = rank_turnover(f, 10)
    cov = coverage_stats(f, fwd)
    mc, _ = max_lib_corr(f, lib_panels)
    gate = abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840
    print(f"{name:<26}{st['ic']:+.4f} {st['icir']:+.3f} {st['hit']:.2f} {st['n']:5d} | "
          f"{st_q['ic']:+.4f} {st_q['icir']:+.3f} | {turn:6.2f} {cov['coverage_dates_ge8']:6.2f} | {mc:.3f} {'PASS' if gate else ''}")

print("\nPer-year h10 IC:")
for name in cands:
    f = cands[name]
    ic = daily_ic(f, forward_ret(close, 10))
    out = []
    for yr in range(2028, 2032):
        sub = ic.loc[ic.index.year == yr]
        st = ic_stats(sub, 10)
        out.append(f"{yr}:{st['ic']:+.3f}(n={st['n']})")
    print(f"{name:<26} " + "  ".join(out))