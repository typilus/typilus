#/usr/bin/env python3

import sys

if len(sys.argv) <= 1:
    print("Usage: (path to repo list)+")
    sys.exit(-1)

OPATH = "uniqTypedPrjs.txt"    
lines = []
for ipath in sys.argv[1:]:
    try:
        with open(ipath, encoding="utf8") as f:
            lines.extend(f.read().splitlines())
    except IOError:
        print(f"Cannot open file: {ipath}.")
        continue

with open(OPATH, "w", encoding="utf8") as f:
    f.writelines(line + "\n" for line in set(lines))
