import pickle
panel = pickle.load(open("scripts/panel_cache.pkl","rb"))
C = panel["close"]
print("panel close dates:", C.index.min().date(), "->", C.index.max().date(), "rows:", C.shape[0])
print(C.tail(2).round(3).T)