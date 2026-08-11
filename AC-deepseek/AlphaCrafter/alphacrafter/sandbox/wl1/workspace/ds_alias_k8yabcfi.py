python - <<'EOF'
path = "scripts/miner1_20270430_explore_batch1.py"
src = open(path).read()
old = """        elif fid == "corr_ret_vol_60":
            vv = V.pct_change()
            fdf[a] = r.rolling(60).corr(vv)"""
new = """        elif fid == "corr_ret_vol_60":
            vv = v.pct_change()
            fdf[a] = r.rolling(60).corr(vv)"""
assert old in src
src = src.replace(old, new)
open(path, "w").write(src)
print("patched")
EOF