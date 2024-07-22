import lorem
import pickle
import sys

N = int(sys.argv[1])

data = [lorem.paragraph() for _ in range(N)]

with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

with open("output.pkl", "rb") as f:
    print(pickle.load(f))
