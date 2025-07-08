import numpy as np
import random

def generate_array(N, X):
    min_required = 4 * X - 1
    if N < min_required:
        raise ValueError("N too small for X separated groups")

    arr = [0] * N
    spots = list(range(N - 2))
    random.shuffle(spots)
    used = set()
    count = 0
    for i in spots:
        if any(j in used for j in range(i - 1, i + 4)):
            continue
        for j in range(i, i + 3): arr[j] = 1; used.add(j)
        count += 1
        if count == X: break
    return arr

print(generate_array(100,25))
