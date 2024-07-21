import lorem
import sys
import json

N = int(sys.argv[1])

data = [
    {"article": lorem.paragraph(), "other_key": "b", "other_int": 1} for _ in range(N)
]

with open("data.json", "w") as f:
    json.dump(data, f, indent=4)
