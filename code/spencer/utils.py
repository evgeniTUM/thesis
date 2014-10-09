# see http://en.wikipedia.org/wiki/Dynamic_time_warping
# see http://en.wikipedia.org/wiki/Longest_common_subsequence_problem

import numpy as np


def euclidean_dist(x,y):
    temp = x-y
    return np.sqrt(np.dot(temp.T, temp))

def dtw(s, t, dist=None):
    n = len(s)
    m = len(t)

    # use eculidean distance if no function specified
    if dist is None:
#        dist = lambda x, y: np.linalg.norm(x-y)
        dist = euclidean_dist # linalg.norm too slow :(

    DTW = np.ndarray((n+1, m+1))

    for i in range(1, n+1):
        DTW[i, 0] = float("inf")
    for i in range(1, m+1):
        DTW[0, i] = float("inf")

    DTW[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist(s[i-1], t[j-1])
            DTW[i, j] = cost + min(DTW[i-1, j],
                                   DTW[i, j-1],
                                   DTW[i-1, j-1])

    return DTW[n, m]


def lcs(s, t):
    n = len(s)
    m = len(t)

    LCS = np.ndarray((n+1, m+1))

    for i in range(n+1):
        LCS[i, 0] = 0.0
    for i in range(m+1):
        LCS[0, i] = 0.0

    for i in range(1, n+1):
        for j in range(1, m+1):
            if s[i-1] == t[j-1]:
                LCS[i,j] = LCS[i-1,j-1] + 1
            else:
                LCS[i, j] = max(LCS[i, j-1], LCS[i-1, j])

    return LCS[n, m]
