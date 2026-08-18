
with open('memory.txt') as f:
    lines = f.readlines()
print("total lines:", len(lines))
for i in range(84, min(105, len(lines))):
    print(f"--- line {i+1} ---")
    print(lines[i][:800])
