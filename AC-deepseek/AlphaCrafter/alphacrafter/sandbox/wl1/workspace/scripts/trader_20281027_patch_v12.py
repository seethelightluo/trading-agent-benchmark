"""Trader patch 2028-10-27: sync ensemble docstring + v12 rank-2 momentum cap."""
src = open('strategy.py').read()

# --- 1. Ensemble description in docstring (weight-only change per Screener) ---
old_ens = ("Ensemble (2028-10-13): vol_of_vol20x60 (.25,+) | vix_beta_cond_60x20 (.24,-)\n"
           "| mom_120d_skip5 (.22,+) | miner2_20260715_nclv_1d (.16,+)\n"
           "| miner2_20260715_rev_2d (.13,+).")
new_ens = ("Ensemble (2028-10-27): vol_of_vol20x60 (.25,+) | mom_120d_skip5 (.24,+)\n"
           "| vix_beta_cond_60x20 (.22,-) | miner2_20260715_nclv_1d (.15,+)\n"
           "| miner2_20260715_rev_2d (.14,+).")
assert old_ens in src, "ensemble docstring block not found"
src = src.replace(old_ens, new_ens)

# --- 2. v12 docstring paragraph appended to the v11 section ---
old_v11_tail = ("deliberately excluded (defensive sleeve). Factor- and trend-agnostic like v9.\n"
                "\"\"\"")
new_v11_tail = ("deliberately excluded (defensive sleeve). Factor- and trend-agnostic like v9.\n\n"
                "v12 (2028-10-27): extend the v10 momentum cap to the top-2 momentum names.\n"
                "Block 1013-1027: SOX was the rank-2 momentum name and delivered -12.5% on\n"
                "7.4% weight - the 9th distinct momentum top-pick whipsaw block in 10\n"
                "(Screener also flags XAU/BTC as top-4 trap names in 20d downtrends). Any of\n"
                "the top-2 momentum names (rank >= .86 of 15) is now capped at GUARD_CAP\n"
                "regardless of MA state, matching the empirical 9/10 top-pick reversal rate.\n"
                "\"\"\"")
assert old_v11_tail in src, "v11 docstring tail not found"
src = src.replace(old_v11_tail, new_v11_tail)

# --- 3. v10 docstring paragraph reference ---
old_v10_ref = "v10 caps the single top momentum name (rank >= MOM_TOP_RANK_STRICT) at"
new_v10_ref = "v10/v12 cap the top-2 momentum names (rank >= MOM_TOP2_RANK) at"
assert old_v10_ref in src, "v10 docstring ref not found"
src = src.replace(old_v10_ref, new_v10_ref)

# --- 4. Constant rename/retune: top-1 -> top-2 ---
old_const = "MOM_TOP_RANK_STRICT = 0.95  # v10: top-1 momentum name (rank >= .95 of 15)"
new_const = "MOM_TOP2_RANK = 0.86  # v12: top-2 momentum names (rank >= .86 of 15)"
assert old_const in src, "constant line not found"
src = src.replace(old_const, new_const)

# --- 5. _ma_guard docstring ---
old_guard_doc = ("    v10: the single top momentum name (rank >= MOM_TOP_RANK_STRICT) is capped\n"
                 "    at GUARD_CAP regardless of MA state, closing the above-MA20 hole that WTI\n"
                 "    exploited in block 0331-0414 (top momentum name, above MA20, -21% crash on\n"
                 "    8.9% weight). Excess is redistributed proportionally to remaining names.")
new_guard_doc = ("    v10/v12: the top-2 momentum names (rank >= MOM_TOP2_RANK) are capped at\n"
                 "    GUARD_CAP regardless of MA state. v10 closed the above-MA20 hole that WTI\n"
                 "    exploited in block 0331-0414 (top momentum name, above MA20, -21% crash on\n"
                 "    8.9% weight); v12 (2028-10-27) extended to rank-2 after SOX delivered\n"
                 "    -12.5% on 7.4% weight (9th top-pick whipsaw block in 10). Excess is\n"
                 "    redistributed proportionally to remaining names.")
assert old_guard_doc in src, "ma_guard docstring not found"
src = src.replace(old_guard_doc, new_guard_doc)

# --- 6. _ma_guard condition: extend strict cap to top-2 ---
old_cond = ("            (mom_rank[a] >= MOM_TOP_RANK and a in below) or\n"
            "            (mom_rank[a] >= MOM_TOP_RANK_STRICT)\n")
new_cond = ("            (mom_rank[a] >= MOM_TOP_RANK and a in below) or\n"
            "            (mom_rank[a] >= MOM_TOP2_RANK)  # v12: top-2, MA-agnostic\n")
assert old_cond in src, "ma_guard condition not found"
src = src.replace(old_cond, new_cond)

assert "MOM_TOP_RANK_STRICT" not in src, "old constant still referenced"
open('strategy.py', 'w').write(src)
print("PATCH OK")
