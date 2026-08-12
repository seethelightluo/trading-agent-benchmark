
lines = open('memory.txt').readlines()
for i in range(90, len(lines)):
    print(i+1, lines[i].rstrip()[:600])
    print()
