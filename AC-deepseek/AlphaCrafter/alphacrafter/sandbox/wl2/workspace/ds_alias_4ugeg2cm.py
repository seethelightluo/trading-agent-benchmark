with open('memory.txt') as f:
    lines = f.readlines()
print("total lines:", len(lines))
print("".join(lines[-4:]))