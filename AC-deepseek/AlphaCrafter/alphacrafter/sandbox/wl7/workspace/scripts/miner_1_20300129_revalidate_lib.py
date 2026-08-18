"""miner_1 revalidation of active factor library at 2030-01-28 (visible date)."""
import sys
sys.path.insert(0, "scripts")
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel)

END = "2030-01-28"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()

lib_panels = library_panel(close, macro)
print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}")

names = ["rel_mom_20d_skip5", "beta_ew_60d", "corr_ew_60", "downside_vol_ratio_20",
         "kurt_20d_skip5", "max_ret_20d", "dxy_beta_cond_60x20", "eurusd_beta_cond_60x20"]

rows = []
for name in names:
    f = lib_panels[name]
    fwd = forward_ret(close, 10)
    ic = daily_ic(f, fwd)
    st = ic_stats(ic, 10)
    cov = coverage_stats(f, fwd)
    turn = rank_turnover(f, 10)
    # last 500 days (approx 2y) and last 250 days (1y) for drift
    f_r = f.tail(500)
    ic_r = daily_ic(f_r, forward_ret(close, 10).reindex(f_r.index))
    st_r = ic_stats(ic_r, 10)
    f_q = f.tail(250)
    ic_q = daily_ic(f_q, forward_ret(close, 10).reindex(f_q.index))
    st_q = ic_stats(ic_q, 10)
    rows.append(dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                     ic_r=st_r["ic"], icir_r=st_r["icir"], n_r=st_r["n"],
                     ic_q=st_q["ic"], icir_q=st_q["icir"], n_q=st_q["n"],
                     covAD=cov["coverage_asset_days"], covD8=cov["coverage_dates_ge8"], turn=turn))

print(f"\n{'factor':26s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} | {'IC_r':>7s} {'ICIR_r':>7s} {'n_r':>4s} | {'IC_q':>7s} {'ICIR_q':>7s} {'n_q':>4s} | {'covAD':>6s} {'covD8':>5s} {'turn':>6s}")
for r in rows:
    print(f"{r['name']:26s} {r['ic']:7.4f} {r['icir']:7.4f} {r['hit']:5.2f} {r['n']:5d} | "
          f"{r['ic_r']:7.4f} {r['icir_r']:7.4f} {r['n_r']:4d} | {r['ic_q']:7.4f} {r['icir_q']:7.4f} {r['n_q']:4d} | "
          f"{r['covAD']:6.3f} {r['covD8']:5.3f} {r['turn']:6.3f}")

# Gate check
print("\nGate: abs(IC)>=0.0070 and abs(ICIR)>=0.0840 (full-window)")
for r in rows:
    ok = abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840
    flag = "PASS" if ok else "FAIL"
    print(f"{r['name']:26s} {flag}  IC={r['ic']:.4f} ICIR={r['icir']:.4f}")
