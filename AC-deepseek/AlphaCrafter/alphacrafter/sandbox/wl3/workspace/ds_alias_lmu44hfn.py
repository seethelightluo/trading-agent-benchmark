import os, glob
for root, dirs, files in os.walk('..'):
    depth = root.count(os.sep) - 1
    if depth > 2: continue
    print(root, "->", sorted(files)[:8], "dirs:", sorted(dirs)[:8])
