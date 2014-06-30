# see http://en.wikipedia.org/wiki/Dynamic_time_warping
import numpy as np


def dtw(s, t, dist=None):
    n = len(s)
    m = len(t)

    if dist is None:
        dist = lambda x, y: np.linalg.norm(x-y)

    DTW = np.ndarray((n+1, m+1))

    for i in range(n):
        DTW[i, 0] = float("inf")
    for i in range(m):
        DTW[0, i] = float("inf")

    DTW[0, 0] = 0

    for i in range(n):
        for j in range(m):
            cost = dist(s[i], t[j])
            DTW[i+1, j+1] = cost + min(DTW[i, j+1],
                                       DTW[i+1, j],
                                       DTW[i, j])

    return DTW[n, m]
