"""miner_1 revalidation of active factor library at 2032-11-02 (visible through 11-01).
Also computes max_abs_library_correlation provenance for each active factor."""
import sys, json
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2032-11-01"
close = ms.load_close(END)
macro = ms.load_macro(END)
lib = ms.library_panel(close, macro)
fwd = ms.forward_ret(close, 10)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

rows = []
for name in lib:
    f = lib[name]
    ic = ms.daily_ic(f, fwd); st = ms.ic_stats(ic, 10)
    ic_r = ms.daily_ic(f.tail(500), ms.forward_ret(close, 10).reindex(f.tail(500).index))
    st_r = ms.ic_stats(ic_r, 10)
    ic_q = ms.daily_ic(f.tail(250), ms.forward_ret(close, 10).reindex(f.tail(250).index))
    st_q = ms.ic_stats(ic_q, 10)
    cov = ms.coverage_stats(f, fwd); turn = ms.rank_turnover(f, 10)
    best, pairs = ms.max_lib_corr(f.tail(500), lib)
    gate = abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE
    rows.append(dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                     ic_r=st_r["ic"], icir_r=st_r["icir"], n_r=st_r["n"],
                     ic_q=st_q["ic"], icir_q=st_q["icir"], n_q=st_q["n"],
                     covAD=cov["coverage_asset_days"], covD8=cov["coverage_dates_ge8"],
                     turn=turn, max_abs_lib_corr=round(best, 4), gate=bool(gate)))

print("\nfactor                     IC10     ICIR10   hit    n | IC_r   ICIR_r |  IC_q  ICIR_q | covAD  covD8  turn  maxRho  gate")
for r in rows:
    print(f"{r['name']:26s} {r['ic']:+.4f} {r['icir']:+.3f} {r['hit']:.3f} {r['n']:5d} | "
          f"{r['ic_r']:+.4f} {r['icir_r']:+.3f} | {r['ic_q']:+.4f} {r['icir_q']:+.3f} | "
          f"{r['covAD']:.3f} {r['covD8']:.3f} {r['turn']:.2f} {r['max_abs_lib_corr']:.3f}  {'PASS' if r['gate'] else 'FAIL'}")

json.dump(rows, open("scripts/miner1_20321102_revalidation.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner1_20321102_revalidation.json")