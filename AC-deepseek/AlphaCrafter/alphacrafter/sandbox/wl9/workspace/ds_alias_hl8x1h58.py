lines = open('scripts/revalidate_all.py').read().splitlines()
for i, l in enumerate(lines[:100], 1):
    print(f"{i:3d} {l}")