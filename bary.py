# Nicolas Bolle 11/19/21ish
# Implementing Wasserstein barycenter stuff on Google Quick Draw! data, using the Sinkhorn divergence

# Some references:
# https://arxiv.org/abs/1306.0895
# https://arxiv.org/abs/1310.4375
# https://arxiv.org/abs/1805.11897

# This file is for standalone methods and functions


# Imports
import numpy as np


# Compute the Sinkhorn divergence between histograms r and c, using a distance matrix M and parameter l=lambda
# For more user-friendly computations, when they only have M and lambda
# FIXME: reduce sizes of M and K based on supports of p1, p2?
def sinkhorn(r,c,M,l):
    K = np.exp(-l*M)
    return sinkhorn_mk(r,c,M,K)


# Compute the Sinkhorn divergence between histograms r and c, using relevant matrices M and K
# For more efficient computations, when K is precomputed
def sinkhorn_mk(r,c,M,K):
    # FIXME: using a fixed number of iterations for now, add a stopping condition
    iters = 1000
    x = np.sum(r) * np.ones(len(r)) / len(r)
    R = np.diag(np.power(r,-1)) @ K
    
    for i in range(iters):
        x = R @ (c * np.power(np.transpose(K) @ np.power(x,-1), -1))
    
    u = np.power(x,-1)
    v = c * np.power(np.transpose(K) @ u, -1)
    return np.sum(u * (np.multiply(K,M) @ v))