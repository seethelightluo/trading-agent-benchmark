with open('scripts/miner_3_20341109_explore.py','a') as f:
    f.write("""
lib_paths = [p for p in lib_paths if 'bak' not in p and 'evicted' not in p and 'rejected' not in p and 'ensemble' not in p and 'deprecated' not in p]
print("library files:", len(lib_paths), flush=True)
# Self-check correlation vs library proxies recomputable from price panel (mom_10d_skip5 as the
# canonical library proxy). Exact rho against all library signal artifacts is the post-Miner gate.
mom10 = zcross(panel(lambda df: df['close'].pct_change(5) / df['close'].shift(15) - 1.0))
for lab, fac in cands.items():
    r = spearman_panel_rho(fac, mom10)
    print(f"  {lab}: rho_vs_mom10_proxy={r:+.3f}", flush=True)
print("done", flush=True)
""")
print("appended")